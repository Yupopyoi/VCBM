using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// Offline RVC v2 F0 WAV -> WAV converter for Unity/Sentis.
///
/// Runtime models:
///   1) ContentVec/HuBERT ONNX: mono 16 kHz waveform -> [1, T, 768] features
///   2) Sentis-compatible RMVPE ONNX: log-mel [1,128,224] -> salience [1,224,360]
///   3) Optional RVC IVF index (.bytes): raw HuBERT retrieval, nprobe=1 / k=8
///   4) RVC v2 F0 generator ONNX: phone/phone_lengths/pitch/pitchf/sid/rnd -> audio
///
/// The generator itself remains fixed at 2 seconds (= 200 RVC frames), but neighboring
/// windows overlap and are overlap-added so hard 2-second boundaries are removed.
/// /// </summary>
[RequireComponent(typeof(AudioSource))]
public sealed class RvcSentisWavConverter : MonoBehaviour
{
    private const int ContentSampleRate = 16000;
    private const int RvcFramesPerSecond = 100;
    private const int PhoneChannels = 768;
    private const int LatentChannels = 192;

    // RVC official RMVPE front-end parameters.
    private const int RmvpeFftSize = 1024;
    private const int RmvpeHopLength = 160;
    private const int RmvpeMelBins = 128;
    private const int RmvpePitchBins = 360;
    private const int RmvpePaddedFrames = 224;
    private const float RmvpeMelFMin = 30.0f;
    private const float RmvpeMelFMax = 8000.0f;
    private const float RmvpeMelClamp = 1e-5f;

    [Header("Input")]
    [Tooltip("Project に取り込んだ変換元 WAV を設定します。")]
    [SerializeField] private AudioClip inputClip;

    [Header("ONNX models")]
    [Tooltip("16 kHz mono waveform -> ContentVec/HuBERT 768-dim features")]
    [SerializeField] private ModelAsset contentVecModelAsset;

    [Tooltip("Sentis-compatible RMVPE ONNX (mel [1,128,224] -> hidden [1,224,360])")]
    [SerializeField] private ModelAsset rmvpeModelAsset;

    [Tooltip("Optional: FAISS .index をUnity用に変換した .bytes。未設定または Index Rate=0 なら検索を使いません。")]
    [SerializeField] private TextAsset retrievalIndexAsset;

    [Tooltip("RVC v2 F0 generator ONNX")]
    [SerializeField] private ModelAsset rvcModelAsset;

    [Header("ContentVec / HuBERT ONNX I/O")]
    [SerializeField] private string contentInputName = "audio";
    [SerializeField] private string contentOutputName = "features";

    public enum FeatureLayout
    {
        TimeChannels, // [1, T, 768]
        ChannelsTime  // [1, 768, T]
    }

    [SerializeField] private FeatureLayout contentOutputLayout = FeatureLayout.TimeChannels;

    [Header("RMVPE ONNX I/O")]
    [SerializeField] private string rmvpeInputName = "mel";
    [SerializeField] private string rmvpeOutputName = "hidden";

    [Header("RVC ONNX I/O")]
    [SerializeField] private string phoneInputName = "phone";
    [SerializeField] private string phoneLengthsInputName = "phone_lengths";
    [SerializeField] private string pitchInputName = "pitch";
    [SerializeField] private string pitchfInputName = "pitchf";
    [SerializeField] private string speakerInputName = "sid"; // official export may call this "ds"
    [SerializeField] private string noiseInputName = "rnd";
    [SerializeField] private string audioOutputName = "audio";

    [Header("RVC settings")]
    [SerializeField] private int speakerId = 0;
    [SerializeField] private int outputSampleRate = 40000;
    [Range(-24, 24)]
    [SerializeField] private int transposeSemitones = 0;

    [Tooltip("RVC ONNX export時の固定フレーム数。添付のSmokeTestと同じ200を既定値にしています。")]
    [Min(1)]
    [SerializeField] private int framesPerChunk = 200;

    [Header("Chunk stitching")]
    [Tooltip("2秒固定窓の重なり。0.5秒なら 0.0-2.0, 1.5-3.5, ... のように処理します。")]
    [Range(0.0f, 1.0f)]
    [SerializeField] private float chunkOverlapSeconds = 0.5f;

    [Header("Retrieval / Protect")]
    [Tooltip("WebUIのIndex Rate相当。0ならindex検索を完全に無効化します。")]
    [Range(0.0f, 1.0f)]
    [SerializeField] private float indexRate = 0.75f;

    [Tooltip("WebUIのProtect相当。0.5で無効、0に近いほど無声音を元HuBERT特徴で強く保護します。")]
    [Range(0.0f, 0.5f)]
    [SerializeField] private float protect = 0.33f;

    [Tooltip("Index検索で1 Unity frameあたり何個のHuBERT frameを処理するか。音質には影響しません。")]
    [Min(1)]
    [SerializeField] private int retrievalQueriesPerFrame = 8;

    [Header("RMVPE / F0")]
    [SerializeField] private float f0MinHz = 50.0f;
    [SerializeField] private float f0MaxHz = 1100.0f;

    [Tooltip("RVC公式RMVPEのinfer_from_audioで使われる既定値は0.03。")]
    [Range(0.0f, 1.0f)]
    [SerializeField] private float rmvpeVoicedThreshold = 0.03f;

    [Header("Sentis")]
    [SerializeField] private BackendType backend = BackendType.GPUCompute;
    [Min(1)]
    [SerializeField] private int layersPerFrame = 30;

    [Header("Output")]
    [SerializeField] private string outputFileName = "rvc_converted.wav";
    [SerializeField] private bool normalizeOutput = true;
    [SerializeField] private bool playWhenFinished = true;

    private AudioSource audioSource;
    private Worker contentWorker;
    private Worker rmvpeWorker;
    private Worker rvcWorker;
    private RvcIvfIndex retrievalIndex;
    private bool running;

    private float[] rmvpeHannWindow;
    private float[] rmvpeMelBasis;

    /// <summary>
    /// Compact Unity-side representation of an RVC FAISS IVF-Flat index.
    ///
    /// Binary layout (little-endian), generated by export_rvc_index_unity.py:
    ///   4 bytes  : "RVCI"
    ///   int32    : version (=1)
    ///   int32    : dimension
    ///   int32    : nlist
    ///   int32    : vectorCount
    ///   float32  : centroids[nlist, dimension]
    ///   int32    : offsets[nlist + 1]
    ///   float32  : vectors[vectorCount, dimension], grouped by IVF list
    ///
    /// Search reproduces the important RVC WebUI behavior:
    /// nprobe=1, k=8, squared-L2 distance, inverse-distance-squared weights.
    /// </summary>
    private sealed class RvcIvfIndex
    {
        private const int Version = 1;
        private const int K = 8;

        public int Dimension { get; private set; }
        public int ListCount { get; private set; }
        public int VectorCount { get; private set; }

        private float[] centroids;
        private int[] offsets;
        private float[] vectors;

        public static RvcIvfIndex Load(byte[] bytes)
        {
            if (bytes == null || bytes.Length < 20)
                throw new InvalidDataException("RVC index bytes are empty.");

            var result = new RvcIvfIndex();

            using var stream = new MemoryStream(bytes, false);
            using var reader = new BinaryReader(stream);

            string magic = new string(reader.ReadChars(4));
            if (magic != "RVCI")
            {
                throw new InvalidDataException(
                    $"Invalid RVC index magic: '{magic}'"
                );
            }

            int version = reader.ReadInt32();
            if (version != Version)
            {
                throw new InvalidDataException(
                    $"Unsupported RVC index version: {version}"
                );
            }

            result.Dimension = reader.ReadInt32();
            result.ListCount = reader.ReadInt32();
            result.VectorCount = reader.ReadInt32();

            if (result.Dimension <= 0 ||
                result.ListCount <= 0 ||
                result.VectorCount <= 0)
            {
                throw new InvalidDataException(
                    "Invalid RVC index dimensions."
                );
            }

            long centroidFloats =
                (long)result.ListCount * result.Dimension;
            long vectorFloats =
                (long)result.VectorCount * result.Dimension;

            if (centroidFloats > int.MaxValue ||
                vectorFloats > int.MaxValue)
            {
                throw new InvalidDataException(
                    "RVC index is too large for this loader."
                );
            }

            result.centroids =
                new float[(int)centroidFloats];

            for (int i = 0; i < result.centroids.Length; i++)
                result.centroids[i] = reader.ReadSingle();

            result.offsets =
                new int[result.ListCount + 1];

            for (int i = 0; i < result.offsets.Length; i++)
                result.offsets[i] = reader.ReadInt32();

            result.vectors =
                new float[(int)vectorFloats];

            for (int i = 0; i < result.vectors.Length; i++)
                result.vectors[i] = reader.ReadSingle();

            if (result.offsets[0] != 0 ||
                result.offsets[result.ListCount] != result.VectorCount)
            {
                throw new InvalidDataException(
                    "RVC index offsets are invalid."
                );
            }

            if (stream.Position != stream.Length)
            {
                Debug.LogWarning(
                    $"[RVC/Index] trailing bytes: " +
                    $"{stream.Length - stream.Position}"
                );
            }

            return result;
        }

        public void SearchWeightedK8(
            float[] query,
            int queryOffset,
            float[] destination,
            int destinationOffset
        )
        {
            if (queryOffset < 0 ||
                queryOffset + Dimension > query.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(queryOffset)
                );
            }

            int bestList = FindNearestCentroid(
                query,
                queryOffset
            );

            int begin = offsets[bestList];
            int end = offsets[bestList + 1];

            if (begin >= end)
            {
                Array.Copy(
                    query,
                    queryOffset,
                    destination,
                    destinationOffset,
                    Dimension
                );
                return;
            }

            float[] bestDistances = new float[K];
            int[] bestIndices = new int[K];

            for (int k = 0; k < K; k++)
            {
                bestDistances[k] = float.PositiveInfinity;
                bestIndices[k] = -1;
            }

            for (int vectorIndex = begin;
                 vectorIndex < end;
                 vectorIndex++)
            {
                int vectorOffset =
                    vectorIndex * Dimension;

                float distance =
                    SquaredL2(
                        query,
                        queryOffset,
                        vectors,
                        vectorOffset,
                        Dimension
                    );

                if (distance >= bestDistances[K - 1])
                    continue;

                int insert = K - 1;

                while (insert > 0 &&
                       distance < bestDistances[insert - 1])
                {
                    bestDistances[insert] =
                        bestDistances[insert - 1];
                    bestIndices[insert] =
                        bestIndices[insert - 1];
                    insert--;
                }

                bestDistances[insert] = distance;
                bestIndices[insert] = vectorIndex;
            }

            double weightTotal = 0.0;
            double[] weights = new double[K];

            for (int k = 0; k < K; k++)
            {
                if (bestIndices[k] < 0)
                    continue;

                // Official RVC:
                //   weight = np.square(1 / score)
                // Clamp only protects the exact-zero numerical corner case.
                double d =
                    Math.Max(bestDistances[k], 1e-12);

                double inv = 1.0 / d;
                double w = inv * inv;

                weights[k] = w;
                weightTotal += w;
            }

            if (weightTotal <= 0.0 ||
                double.IsNaN(weightTotal) ||
                double.IsInfinity(weightTotal))
            {
                Array.Copy(
                    query,
                    queryOffset,
                    destination,
                    destinationOffset,
                    Dimension
                );
                return;
            }

            for (int c = 0; c < Dimension; c++)
            {
                double value = 0.0;

                for (int k = 0; k < K; k++)
                {
                    int vectorIndex = bestIndices[k];
                    if (vectorIndex < 0)
                        continue;

                    double normalizedWeight =
                        weights[k] / weightTotal;

                    value +=
                        vectors[
                            vectorIndex * Dimension + c
                        ] * normalizedWeight;
                }

                destination[destinationOffset + c] =
                    (float)value;
            }
        }

        private int FindNearestCentroid(
            float[] query,
            int queryOffset
        )
        {
            int best = 0;
            float bestDistance =
                float.PositiveInfinity;

            for (int list = 0;
                 list < ListCount;
                 list++)
            {
                float distance =
                    SquaredL2(
                        query,
                        queryOffset,
                        centroids,
                        list * Dimension,
                        Dimension
                    );

                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = list;
                }
            }

            return best;
        }

        private static float SquaredL2(
            float[] a,
            int aOffset,
            float[] b,
            int bOffset,
            int length
        )
        {
            double sum = 0.0;

            for (int i = 0; i < length; i++)
            {
                double d =
                    a[aOffset + i] -
                    b[bOffset + i];

                sum += d * d;
            }

            return (float)sum;
        }
    }

    private sealed class PhoneFeatureBundle
    {
        public float[] Original;
        public float[] Processed;
        public int RawFrames;
    }

    private void Start()
    {
        audioSource = GetComponent<AudioSource>();

        if (!ValidateInspector())
            return;

        StartCoroutine(Convert());
    }

    [ContextMenu("Convert WAV with RVC")]
    public void ConvertFromContextMenu()
    {
        if (!Application.isPlaying)
        {
            Debug.LogWarning("[RVC] Play Modeで実行してください。");
            return;
        }

        if (!running)
            StartCoroutine(Convert());
    }

    private bool ValidateInspector()
    {
        if (inputClip == null)
        {
            Debug.LogError("[RVC] Input Clip が未設定です。");
            return false;
        }

        if (contentVecModelAsset == null)
        {
            Debug.LogError("[RVC] ContentVec/HuBERT ONNX が未設定です。");
            return false;
        }

        if (rmvpeModelAsset == null)
        {
            Debug.LogError("[RVC] RMVPE ONNX が未設定です。");
            return false;
        }

        if (rvcModelAsset == null)
        {
            Debug.LogError("[RVC] RVC ONNX が未設定です。");
            return false;
        }

        if (framesPerChunk != 200)
        {
            Debug.LogError(
                "[RVC] 現在の固定ONNX構成では framesPerChunk は200にしてください。"
            );
            return false;
        }

        if (outputSampleRate <= 0)
        {
            Debug.LogError("[RVC] outputSampleRate が不正です。");
            return false;
        }

        float chunkSeconds = framesPerChunk / (float)RvcFramesPerSecond;
        if (chunkOverlapSeconds < 0.0f || chunkOverlapSeconds >= chunkSeconds)
        {
            Debug.LogError(
                $"[RVC] chunkOverlapSeconds は 0 以上 {chunkSeconds:F2} 秒未満にしてください。"
            );
            return false;
        }

        if (indexRate > 0.0f && retrievalIndexAsset == null)
        {
            Debug.LogWarning(
                "[RVC] Index Rate > 0 ですが Retrieval Index (.bytes) が未設定です。 " +
                "今回はindex検索なしで続行します。"
            );
        }

        return true;
    }

    private IEnumerator Convert()
    {
        if (running)
            yield break;

        running = true;

        try
        {
            Debug.Log("[RVC] モデルを読み込みます。");

            Model contentModel = ModelLoader.Load(contentVecModelAsset);
            Model rmvpeModel = ModelLoader.Load(rmvpeModelAsset);
            Model rvcModel = ModelLoader.Load(rvcModelAsset);

            LogModelIO("ContentVec", contentModel);
            LogModelIO("RMVPE", rmvpeModel);
            LogModelIO("RVC", rvcModel);

            contentWorker = new Worker(contentModel, backend);
            rmvpeWorker = new Worker(rmvpeModel, backend);
            rvcWorker = new Worker(rvcModel, backend);

            retrievalIndex = null;
            if (retrievalIndexAsset != null && indexRate > 0.0f)
            {
                retrievalIndex = RvcIvfIndex.Load(
                    retrievalIndexAsset.bytes
                );

                if (retrievalIndex.Dimension != PhoneChannels)
                {
                    throw new InvalidOperationException(
                        $"Index dimension={retrievalIndex.Dimension}, " +
                        $"RVC v2 requires {PhoneChannels}."
                    );
                }

                Debug.Log(
                    $"[RVC/Index] loaded: nlist={retrievalIndex.ListCount}, " +
                    $"vectors={retrievalIndex.VectorCount}, dim={retrievalIndex.Dimension}"
                );
            }

            EnsureRmvpeDspTables();

            float[] sourceMono = ReadMono(inputClip);
            float[] source16k = ResampleLinear(sourceMono, inputClip.frequency, ContentSampleRate);

            int contentSamplesPerChunk =
                Mathf.RoundToInt(ContentSampleRate * (framesPerChunk / (float)RvcFramesPerSecond));

            int outputSamplesPerChunk =
                Mathf.RoundToInt(outputSampleRate * (framesPerChunk / (float)RvcFramesPerSecond));

            int wantedOutputSamples = Mathf.RoundToInt(
                source16k.Length / (float)ContentSampleRate * outputSampleRate
            );

            int overlapSamples16k = Mathf.RoundToInt(
                chunkOverlapSeconds * ContentSampleRate
            );
            int strideSamples16k = contentSamplesPerChunk - overlapSamples16k;

            List<int> chunkStarts = BuildOverlappingChunkStarts(
                source16k.Length,
                contentSamplesPerChunk,
                strideSamples16k
            );

            Debug.Log(
                $"[RVC] Input clip: duration={inputClip.length:F4}s, " +
                $"frequency={inputClip.frequency}, channels={inputClip.channels}, " +
                $"samplesPerChannel={inputClip.samples}, monoSamples={sourceMono.Length}"
            );

            Debug.Log(
                $"[RVC] Resampled: 16kSamples={source16k.Length}, " +
                $"duration={source16k.Length / (float)ContentSampleRate:F4}s, " +
                $"window={contentSamplesPerChunk / (float)ContentSampleRate:F3}s, " +
                $"overlap={chunkOverlapSeconds:F3}s, chunks={chunkStarts.Count}, " +
                $"wantedOutputSamples={wantedOutputSamples}"
            );

            float[] outputSum = new float[wantedOutputSamples];
            float[] outputWeight = new float[wantedOutputSamples];
            var rng = new System.Random(1234567);

            int fadeSamplesOutput = Mathf.RoundToInt(
                chunkOverlapSeconds * outputSampleRate
            );

            for (int chunkIndex = 0; chunkIndex < chunkStarts.Count; chunkIndex++)
            {
                int srcOffset = chunkStarts[chunkIndex];

                Debug.Log(
                    $"[RVC] Chunk {chunkIndex + 1}/{chunkStarts.Count}, " +
                    $"start={srcOffset / (float)ContentSampleRate:F3}s"
                );

                float[] chunk16k = new float[contentSamplesPerChunk];
                int copyCount = Mathf.Min(
                    contentSamplesPerChunk,
                    source16k.Length - srcOffset
                );

                if (copyCount > 0)
                    Array.Copy(source16k, srcOffset, chunk16k, 0, copyCount);

                // 1) ContentVec / HuBERT (+ optional retrieval index)
                PhoneFeatureBundle phoneBundle = null;
                yield return ExtractPhone(
                    chunk16k,
                    result => phoneBundle = result
                );

                if (phoneBundle == null ||
                    phoneBundle.Processed == null ||
                    phoneBundle.Original == null ||
                    phoneBundle.Processed.Length != framesPerChunk * PhoneChannels ||
                    phoneBundle.Original.Length != framesPerChunk * PhoneChannels)
                {
                    throw new InvalidOperationException(
                        "HuBERT/retrieval phone shape mismatch."
                    );
                }

                // 2) RMVPE -> pitch / pitchf
                int[] pitch = null;
                float[] pitchf = null;

                yield return ExtractF0Rmvpe(
                    chunk16k,
                    framesPerChunk,
                    (p, pf) =>
                    {
                        pitch = p;
                        pitchf = pf;
                    }
                );

                if (pitch == null || pitchf == null ||
                    pitch.Length != framesPerChunk ||
                    pitchf.Length != framesPerChunk)
                {
                    throw new InvalidOperationException(
                        $"RMVPE pitch shape mismatch: pitch={pitch?.Length ?? 0}, " +
                        $"pitchf={pitchf?.Length ?? 0}, expected={framesPerChunk}"
                    );
                }

                if (transposeSemitones != 0)
                {
                    float ratio = Mathf.Pow(
                        2.0f,
                        transposeSemitones / 12.0f
                    );

                    for (int i = 0; i < pitchf.Length; i++)
                    {
                        pitchf[i] =
                            pitchf[i] > 0.0f
                                ? pitchf[i] * ratio
                                : 0.0f;
                    }

                    for (int i = 0; i < pitch.Length; i++)
                        pitch[i] = F0ToCoarse(pitchf[i]);
                }

                // WebUI-compatible protect:
                // voiced   : processed features 100%
                // unvoiced : processed*protect + original*(1-protect)
                ApplyProtect(
                    phoneBundle.Processed,
                    phoneBundle.Original,
                    pitchf,
                    protect
                );

                // 3) RVC generator
                float[] chunkOutput = null;

                yield return RunRvc(
                    phoneBundle.Processed,
                    pitch,
                    pitchf,
                    rng,
                    result => chunkOutput = result
                );

                if (chunkOutput == null || chunkOutput.Length == 0)
                    throw new InvalidOperationException("RVC出力が空です。");

                Debug.Log(
                    $"[RVC] Generator raw output: samples={chunkOutput.Length}, " +
                    $"duration={chunkOutput.Length / (float)outputSampleRate:F4}s"
                );

                if (chunkOutput.Length != outputSamplesPerChunk)
                {
                    Debug.LogWarning(
                        $"[RVC] Generator output length={chunkOutput.Length}; " +
                        $"expected={outputSamplesPerChunk}. Resampling this chunk."
                    );

                    chunkOutput = ResampleToLength(
                        chunkOutput,
                        outputSamplesPerChunk
                    );
                }

                int outputStart = Mathf.RoundToInt(
                    srcOffset / (float)ContentSampleRate * outputSampleRate
                );

                AddChunkOverlap(
                    outputSum,
                    outputWeight,
                    chunkOutput,
                    outputStart,
                    fadeSamplesOutput,
                    chunkIndex == 0,
                    chunkIndex == chunkStarts.Count - 1
                );

                yield return null;
            }

            float[] output = FinalizeOverlapAdd(
                outputSum,
                outputWeight
            );

            ValidateAudio(output);

            if (wantedOutputSamples <= 0)
            {
                throw new InvalidOperationException(
                    $"wantedOutputSamples が不正です: {wantedOutputSamples}"
                );
            }

            if (normalizeOutput)
                NormalizeInPlace(output, 0.95f);

            AnalyzeSignal(output, out float outPeak, out float outRms);

            Debug.Log(
                $"[RVC] Final output: samples={output.Length}, " +
                $"sampleRate={outputSampleRate}, " +
                $"duration={output.Length / (float)outputSampleRate:F4}s, " +
                $"peak={outPeak:F6}, rms={outRms:F6}"
            );

            string outputPath = Path.Combine(Application.persistentDataPath, outputFileName);
            WriteMonoPcm16Wav(outputPath, output, outputSampleRate);
            ValidateWrittenWav(outputPath);

            Debug.Log($"[RVC] WAV保存完了: {outputPath}");

            if (playWhenFinished)
            {
                AudioClip converted = AudioClip.Create(
                    "RVC Converted",
                    output.Length,
                    1,
                    outputSampleRate,
                    false
                );

                if (!converted.SetData(output, 0))
                    throw new InvalidOperationException("AudioClip.SetData に失敗しました。");

                audioSource.clip = converted;
                audioSource.Play();
            }
        }
        finally
        {
            contentWorker?.Dispose();
            contentWorker = null;

            rmvpeWorker?.Dispose();
            rmvpeWorker = null;

            retrievalIndex = null;

            rvcWorker?.Dispose();
            rvcWorker = null;

            running = false;
        }
    }

    private IEnumerator ExtractPhone(
        float[] audio16k,
        Action<PhoneFeatureBundle> completed
    )
    {
        Tensor<float> input = null;
        Tensor<float> cpuOutput = null;

        try
        {
            input = new Tensor<float>(
                new TensorShape(1, audio16k.Length),
                audio16k
            );

            contentWorker.SetInput(contentInputName, input);

            yield return ScheduleInSlices(
                contentWorker,
                "ContentVec"
            );

            Tensor<float> output =
                contentWorker.PeekOutput(contentOutputName)
                as Tensor<float>;

            if (output == null)
            {
                throw new InvalidOperationException(
                    $"ContentVec output '{contentOutputName}' を " +
                    "Tensor<float> として取得できません。"
                );
            }

            cpuOutput = output.ReadbackAndClone();
            float[] raw = cpuOutput.DownloadToArray();

            if (raw.Length % PhoneChannels != 0)
            {
                throw new InvalidOperationException(
                    $"ContentVec output length {raw.Length} is not divisible by " +
                    $"{PhoneChannels}."
                );
            }

            int sourceFrames = raw.Length / PhoneChannels;
            float[] originalRaw =
                ToTimeMajorFeatures(raw, sourceFrames);

            float[] processedRaw =
                (float[])originalRaw.Clone();

            if (retrievalIndex != null && indexRate > 0.0f)
            {
                float[] retrieved = null;

                yield return RunRetrievalIndex(
                    originalRaw,
                    sourceFrames,
                    result => retrieved = result
                );

                if (retrieved == null ||
                    retrieved.Length != originalRaw.Length)
                {
                    throw new InvalidOperationException(
                        $"Retrieval output mismatch: got={retrieved?.Length ?? 0}, " +
                        $"expected={originalRaw.Length}"
                    );
                }

                float keepOriginal = 1.0f - indexRate;

                for (int i = 0; i < processedRaw.Length; i++)
                {
                    processedRaw[i] =
                        retrieved[i] * indexRate +
                        originalRaw[i] * keepOriginal;
                }
            }

            // RVC v2's official interpolation is nearest-neighbor x2.
            // 32000 samples -> HuBERT 99 frames -> 198 frames.
            // Our fixed generator is 200 frames, so duplicate x2 first,
            // then pad the final two frames with the last valid feature
            // instead of stretching 99 frames across all 200 positions.
            float[] originalPhone =
                UpsampleRvcV2FeaturesToFixed(
                    originalRaw,
                    sourceFrames,
                    framesPerChunk,
                    PhoneChannels
                );

            float[] processedPhone =
                UpsampleRvcV2FeaturesToFixed(
                    processedRaw,
                    sourceFrames,
                    framesPerChunk,
                    PhoneChannels
                );

            Debug.Log(
                $"[RVC/HuBERT] rawFrames={sourceFrames}, " +
                $"generatorFrames={framesPerChunk}, " +
                $"index={(retrievalIndex != null && indexRate > 0.0f ? indexRate.ToString("F2") : "off")}"
            );

            completed(
                new PhoneFeatureBundle
                {
                    Original = originalPhone,
                    Processed = processedPhone,
                    RawFrames = sourceFrames,
                }
            );
        }
        finally
        {
            cpuOutput?.Dispose();
            input?.Dispose();
        }
    }

    private IEnumerator RunRetrievalIndex(
        float[] rawFeatures,
        int rawFrames,
        Action<float[]> completed
    )
    {
        if (retrievalIndex == null)
        {
            completed((float[])rawFeatures.Clone());
            yield break;
        }

        float[] retrieved =
            new float[rawFeatures.Length];

        int batch =
            Mathf.Max(1, retrievalQueriesPerFrame);

        for (int frame = 0; frame < rawFrames; frame++)
        {
            retrievalIndex.SearchWeightedK8(
                rawFeatures,
                frame * PhoneChannels,
                retrieved,
                frame * PhoneChannels
            );

            if ((frame + 1) % batch == 0)
                yield return null;
        }

        completed(retrieved);
    }

    private float[] ToTimeMajorFeatures(float[] raw, int frames)
    {
        if (contentOutputLayout == FeatureLayout.TimeChannels)
            return raw;

        float[] dst = new float[frames * PhoneChannels];
        for (int c = 0; c < PhoneChannels; c++)
        {
            for (int t = 0; t < frames; t++)
            {
                dst[t * PhoneChannels + c] = raw[c * frames + t];
            }
        }
        return dst;
    }

    private IEnumerator RunRvc(
        float[] phone,
        int[] pitch,
        float[] pitchf,
        System.Random rng,
        Action<float[]> completed
    )
    {
        Tensor<float> phoneTensor = null;
        Tensor<int> phoneLengthsTensor = null;
        Tensor<int> pitchTensor = null;
        Tensor<float> pitchfTensor = null;
        Tensor<int> speakerTensor = null;
        Tensor<float> noiseTensor = null;
        Tensor<float> cpuOutput = null;

        try
        {
            float[] noise = new float[LatentChannels * framesPerChunk];
            for (int i = 0; i < noise.Length; i++)
                noise[i] = NextGaussian(rng);

            phoneTensor = new Tensor<float>(
                new TensorShape(1, framesPerChunk, PhoneChannels),
                phone
            );
            phoneLengthsTensor = new Tensor<int>(new TensorShape(1), new[] { framesPerChunk });
            pitchTensor = new Tensor<int>(new TensorShape(1, framesPerChunk), pitch);
            pitchfTensor = new Tensor<float>(new TensorShape(1, framesPerChunk), pitchf);
            speakerTensor = new Tensor<int>(new TensorShape(1), new[] { speakerId });
            noiseTensor = new Tensor<float>(
                new TensorShape(1, LatentChannels, framesPerChunk),
                noise
            );

            rvcWorker.SetInput(phoneInputName, phoneTensor);
            rvcWorker.SetInput(phoneLengthsInputName, phoneLengthsTensor);
            rvcWorker.SetInput(pitchInputName, pitchTensor);
            rvcWorker.SetInput(pitchfInputName, pitchfTensor);
            rvcWorker.SetInput(speakerInputName, speakerTensor);
            rvcWorker.SetInput(noiseInputName, noiseTensor);

            yield return ScheduleInSlices(rvcWorker, "RVC");

            Tensor<float> output = rvcWorker.PeekOutput(audioOutputName) as Tensor<float>;
            if (output == null)
            {
                throw new InvalidOperationException(
                    $"RVC output '{audioOutputName}' を Tensor<float> として取得できません。"
                );
            }

            cpuOutput = output.ReadbackAndClone();
            completed(cpuOutput.DownloadToArray());
        }
        finally
        {
            cpuOutput?.Dispose();
            noiseTensor?.Dispose();
            speakerTensor?.Dispose();
            pitchfTensor?.Dispose();
            pitchTensor?.Dispose();
            phoneLengthsTensor?.Dispose();
            phoneTensor?.Dispose();
        }
    }

    private IEnumerator ScheduleInSlices(Worker worker, string label)
    {
        IEnumerator schedule = worker.ScheduleIterable();
        int layer = 0;

        while (schedule.MoveNext())
        {
            layer++;
            if (layer % layersPerFrame == 0)
                yield return null;
        }

        Debug.Log($"[RVC] {label}: scheduled {layer} layers");
    }

    /// <summary>
    /// RVC official RMVPE-compatible path:
    /// 16 kHz waveform -> log-mel -> Sentis RMVPE -> 360-bin salience -> F0.
    ///
    /// The RMVPE ONNX used here is the GRU-unrolled fixed model exported as:
    ///   input : mel    [1,128,224]
    ///   output: hidden [1,224,360]
    /// </summary>
    private IEnumerator ExtractF0Rmvpe(
        float[] audio16k,
        int targetFrames,
        Action<int[], float[]> completed
    )
    {
        Tensor<float> melTensor = null;
        Tensor<float> cpuOutput = null;

        try
        {
            float[] mel = BuildRmvpeLogMel(audio16k, out int realMelFrames);

            if (realMelFrames > RmvpePaddedFrames)
            {
                throw new InvalidOperationException(
                    $"RMVPE mel frames={realMelFrames} exceeds fixed ONNX " +
                    $"capacity={RmvpePaddedFrames}."
                );
            }

            melTensor = new Tensor<float>(
                new TensorShape(1, RmvpeMelBins, RmvpePaddedFrames),
                mel
            );

            rmvpeWorker.SetInput(rmvpeInputName, melTensor);

            yield return ScheduleInSlices(rmvpeWorker, "RMVPE");

            Tensor<float> output =
                rmvpeWorker.PeekOutput(rmvpeOutputName) as Tensor<float>;

            if (output == null)
            {
                throw new InvalidOperationException(
                    $"RMVPE output '{rmvpeOutputName}' を " +
                    "Tensor<float> として取得できません。"
                );
            }

            cpuOutput = output.ReadbackAndClone();
            float[] hidden = cpuOutput.DownloadToArray();

            if (hidden.Length % RmvpePitchBins != 0)
            {
                throw new InvalidOperationException(
                    $"RMVPE output length={hidden.Length} is not divisible by " +
                    $"{RmvpePitchBins}."
                );
            }

            int networkFrames = hidden.Length / RmvpePitchBins;

            if (networkFrames < realMelFrames)
            {
                throw new InvalidOperationException(
                    $"RMVPE output frames={networkFrames} < mel frames={realMelFrames}."
                );
            }

            DecodeRmvpe(
                hidden,
                realMelFrames,
                targetFrames,
                out int[] pitch,
                out float[] pitchf
            );

            int voiced = 0;
            float minF0 = float.PositiveInfinity;
            float maxF0 = 0.0f;
            double sumF0 = 0.0;

            for (int i = 0; i < pitchf.Length; i++)
            {
                if (pitchf[i] <= 0.0f)
                    continue;

                voiced++;
                minF0 = Mathf.Min(minF0, pitchf[i]);
                maxF0 = Mathf.Max(maxF0, pitchf[i]);
                sumF0 += pitchf[i];
            }

            float meanF0 = voiced > 0 ? (float)(sumF0 / voiced) : 0.0f;
            if (voiced == 0)
                minF0 = 0.0f;

            Debug.Log(
                $"[RVC/RMVPE] melFrames={realMelFrames}->{RmvpePaddedFrames}, " +
                $"networkFrames={networkFrames}, targetFrames={targetFrames}, " +
                $"voiced={voiced}/{targetFrames}, " +
                $"f0(min/mean/max)={minF0:F2}/{meanF0:F2}/{maxF0:F2} Hz"
            );

            completed(pitch, pitchf);
        }
        finally
        {
            cpuOutput?.Dispose();
            melTensor?.Dispose();
        }
    }

    /// <summary>
    /// Reproduces the RMVPE MelSpectrogram settings used by RVC:
    ///   sr=16000, n_fft=1024, win=1024, hop=160,
    ///   n_mels=128, fmin=30, fmax=8000, HTK mel,
    ///   Slaney area normalization, natural log, clamp=1e-5,
    ///   center=true with reflect padding.
    ///
    /// The real log-mel frames are written first and the remaining time
    /// positions are zero-filled, matching RMVPE.mel2hidden() constant padding.
    /// </summary>
    private float[] BuildRmvpeLogMel(float[] audio16k, out int realFrames)
    {
        if (audio16k == null || audio16k.Length == 0)
            throw new ArgumentException("RMVPE input audio is empty.");

        EnsureRmvpeDspTables();

        int pad = RmvpeFftSize / 2;

        realFrames =
            1 + (audio16k.Length + 2 * pad - RmvpeFftSize) / RmvpeHopLength;

        if (realFrames <= 0)
            throw new InvalidOperationException($"Invalid RMVPE frame count: {realFrames}");

        if (realFrames > RmvpePaddedFrames)
        {
            throw new InvalidOperationException(
                $"This fixed RMVPE ONNX supports at most {RmvpePaddedFrames} " +
                $"mel frames, but input produced {realFrames}."
            );
        }

        // Tensor layout [1, 128, 224] => channel-major, time-last.
        // Remaining frames intentionally stay exactly 0.0f.
        float[] logMel =
            new float[RmvpeMelBins * RmvpePaddedFrames];

        float[] real = new float[RmvpeFftSize];
        float[] imag = new float[RmvpeFftSize];
        float[] magnitude = new float[RmvpeFftSize / 2 + 1];

        for (int frame = 0; frame < realFrames; frame++)
        {
            int sourceStart = frame * RmvpeHopLength - pad;

            for (int n = 0; n < RmvpeFftSize; n++)
            {
                int sourceIndex = ReflectIndex(
                    sourceStart + n,
                    audio16k.Length
                );

                real[n] =
                    audio16k[sourceIndex] * rmvpeHannWindow[n];
                imag[n] = 0.0f;
            }

            FftRadix2InPlace(real, imag);

            for (int bin = 0; bin < magnitude.Length; bin++)
            {
                double re = real[bin];
                double im = imag[bin];
                magnitude[bin] = (float)Math.Sqrt(re * re + im * im);
            }

            for (int mel = 0; mel < RmvpeMelBins; mel++)
            {
                int basisOffset = mel * magnitude.Length;
                double sum = 0.0;

                for (int bin = 0; bin < magnitude.Length; bin++)
                {
                    sum +=
                        rmvpeMelBasis[basisOffset + bin] *
                        magnitude[bin];
                }

                float value = Mathf.Max((float)sum, RmvpeMelClamp);

                logMel[
                    mel * RmvpePaddedFrames + frame
                ] = Mathf.Log(value);
            }
        }

        return logMel;
    }

    private void EnsureRmvpeDspTables()
    {
        if (rmvpeHannWindow == null)
        {
            rmvpeHannWindow = new float[RmvpeFftSize];

            // torch.hann_window(1024) uses periodic=True by default:
            // w[n] = 0.5 - 0.5*cos(2*pi*n/N)
            for (int n = 0; n < RmvpeFftSize; n++)
            {
                rmvpeHannWindow[n] = (float)(
                    0.5 -
                    0.5 * Math.Cos(
                        2.0 * Math.PI * n / RmvpeFftSize
                    )
                );
            }
        }

        if (rmvpeMelBasis == null)
            rmvpeMelBasis = BuildLibrosaHtkMelBasis();
    }

    /// <summary>
    /// Equivalent to:
    /// librosa.filters.mel(
    ///     sr=16000, n_fft=1024, n_mels=128,
    ///     fmin=30, fmax=8000, htk=true
    /// )
    /// with librosa's default norm="slaney".
    /// </summary>
    private static float[] BuildLibrosaHtkMelBasis()
    {
        int frequencyBins = RmvpeFftSize / 2 + 1;
        float[] basis = new float[RmvpeMelBins * frequencyBins];

        double minMel = HzToHtkMel(RmvpeMelFMin);
        double maxMel = HzToHtkMel(RmvpeMelFMax);

        double[] melFrequencies = new double[RmvpeMelBins + 2];

        for (int i = 0; i < melFrequencies.Length; i++)
        {
            double alpha =
                i / (double)(melFrequencies.Length - 1);

            double mel =
                minMel + (maxMel - minMel) * alpha;

            melFrequencies[i] = HtkMelToHz(mel);
        }

        for (int m = 0; m < RmvpeMelBins; m++)
        {
            double left = melFrequencies[m];
            double center = melFrequencies[m + 1];
            double right = melFrequencies[m + 2];

            double leftWidth = center - left;
            double rightWidth = right - center;

            // librosa's default "slaney" area normalization.
            double enorm = 2.0 / (right - left);

            for (int k = 0; k < frequencyBins; k++)
            {
                double hz =
                    k * ContentSampleRate / (double)RmvpeFftSize;

                double lower =
                    (hz - left) / leftWidth;

                double upper =
                    (right - hz) / rightWidth;

                double weight =
                    Math.Max(0.0, Math.Min(lower, upper));

                basis[m * frequencyBins + k] =
                    (float)(weight * enorm);
            }
        }

        return basis;
    }

    private static double HzToHtkMel(double hz)
    {
        return 2595.0 * Math.Log10(1.0 + hz / 700.0);
    }

    private static double HtkMelToHz(double mel)
    {
        return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
    }

    /// <summary>
    /// torch.stft(center=true, pad_mode="reflect") compatible index reflection.
    /// For [a,b,c,d], pad=2 gives [c,b,a,b,c,d,c,b].
    /// </summary>
    private static int ReflectIndex(int index, int length)
    {
        if (length <= 1)
            return 0;

        while (index < 0 || index >= length)
        {
            if (index < 0)
                index = -index;

            if (index >= length)
                index = 2 * length - index - 2;
        }

        return index;
    }

    /// <summary>
    /// In-place forward FFT. 1024 is a power of two.
    /// No normalization is applied, matching torch.stft forward FFT magnitude.
    /// </summary>
    private static void FftRadix2InPlace(float[] real, float[] imag)
    {
        int n = real.Length;

        if (imag.Length != n || (n & (n - 1)) != 0)
            throw new ArgumentException("FFT length must be a power of two.");

        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;

            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;

            j ^= bit;

            if (i < j)
            {
                float tmpReal = real[i];
                real[i] = real[j];
                real[j] = tmpReal;

                float tmpImag = imag[i];
                imag[i] = imag[j];
                imag[j] = tmpImag;
            }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1;
            double angle = -2.0 * Math.PI / len;

            double stepReal = Math.Cos(angle);
            double stepImag = Math.Sin(angle);

            for (int block = 0; block < n; block += len)
            {
                double wReal = 1.0;
                double wImag = 0.0;

                for (int j = 0; j < half; j++)
                {
                    int evenIndex = block + j;
                    int oddIndex = evenIndex + half;

                    double oddReal =
                        real[oddIndex] * wReal -
                        imag[oddIndex] * wImag;

                    double oddImag =
                        real[oddIndex] * wImag +
                        imag[oddIndex] * wReal;

                    double evenReal = real[evenIndex];
                    double evenImag = imag[evenIndex];

                    real[evenIndex] =
                        (float)(evenReal + oddReal);
                    imag[evenIndex] =
                        (float)(evenImag + oddImag);

                    real[oddIndex] =
                        (float)(evenReal - oddReal);
                    imag[oddIndex] =
                        (float)(evenImag - oddImag);

                    double nextWReal =
                        wReal * stepReal -
                        wImag * stepImag;

                    double nextWImag =
                        wReal * stepImag +
                        wImag * stepReal;

                    wReal = nextWReal;
                    wImag = nextWImag;
                }
            }
        }
    }

    /// <summary>
    /// Equivalent to RVC RMVPE.decode()/to_local_average_cents():
    /// argmax -> +-4-bin local weighted average -> cents -> Hz.
    /// The RMVPE output has 360 salience bins.
    /// </summary>
    private void DecodeRmvpe(
        float[] hidden,
        int realMelFrames,
        int targetFrames,
        out int[] pitch,
        out float[] pitchf
    )
    {
        pitch = new int[targetFrames];
        pitchf = new float[targetFrames];

        int availableFrames =
            Mathf.Min(
                realMelFrames,
                hidden.Length / RmvpePitchBins
            );

        for (int frame = 0; frame < targetFrames; frame++)
        {
            if (frame >= availableFrames)
            {
                pitch[frame] = 1;
                pitchf[frame] = 0.0f;
                continue;
            }

            int baseIndex = frame * RmvpePitchBins;

            int center = 0;
            float maxSalience = hidden[baseIndex];

            for (int bin = 1; bin < RmvpePitchBins; bin++)
            {
                float value = hidden[baseIndex + bin];

                if (value > maxSalience)
                {
                    maxSalience = value;
                    center = bin;
                }
            }

            if (maxSalience <= rmvpeVoicedThreshold)
            {
                pitch[frame] = 1;
                pitchf[frame] = 0.0f;
                continue;
            }

            double weightedCents = 0.0;
            double weightSum = 0.0;

            int first = Mathf.Max(0, center - 4);
            int last = Mathf.Min(RmvpePitchBins - 1, center + 4);

            for (int bin = first; bin <= last; bin++)
            {
                double salience = hidden[baseIndex + bin];

                // RVC: cents_mapping = 20 * arange(360) + 1997.3794084376191
                double cents =
                    20.0 * bin + 1997.3794084376191;

                weightedCents += salience * cents;
                weightSum += salience;
            }

            if (weightSum <= 1e-12)
            {
                pitch[frame] = 1;
                pitchf[frame] = 0.0f;
                continue;
            }

            double centsPrediction =
                weightedCents / weightSum;

            float f0 = (float)(
                10.0 *
                Math.Pow(2.0, centsPrediction / 1200.0)
            );

            // RVC converts cents=0 to 10 Hz and then replaces that with 0.
            // We already handled unvoiced frames above, so voiced frames keep f0.
            pitchf[frame] = f0;
            pitch[frame] = F0ToCoarse(f0);
        }
    }

    private int F0ToCoarse(float f0)
    {
        if (f0 <= 0.0f)
            return 1;

        const float f0Bin = 256.0f;
        float melMin = 1127.0f * Mathf.Log(1.0f + f0MinHz / 700.0f);
        float melMax = 1127.0f * Mathf.Log(1.0f + f0MaxHz / 700.0f);
        float mel = 1127.0f * Mathf.Log(1.0f + f0 / 700.0f);

        float coarse = (mel - melMin) * (f0Bin - 2.0f) / (melMax - melMin) + 1.0f;
        return Mathf.Clamp(Mathf.RoundToInt(coarse), 1, 255);
    }

    private static float[] ReadMono(AudioClip clip)
    {
        int channels = clip.channels;
        float[] interleaved = new float[clip.samples * channels];
        if (!clip.GetData(interleaved, 0))
            throw new InvalidOperationException("AudioClip.GetData に失敗しました。");

        if (channels == 1)
            return interleaved;

        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            double sum = 0.0;
            int baseIndex = i * channels;
            for (int c = 0; c < channels; c++)
                sum += interleaved[baseIndex + c];
            mono[i] = (float)(sum / channels);
        }
        return mono;
    }

    private static float[] ResampleLinear(float[] source, int sourceRate, int destinationRate)
    {
        if (sourceRate == destinationRate)
            return (float[])source.Clone();

        // IMPORTANT:
        // source.Length * destinationRate を int 同士で計算すると、
        // 数秒程度の音声でも Int32 の上限を超えてオーバーフローします。
        //
        // 例:
        // 44.1 kHz × 5 sec = 220500 samples
        // 220500 * 16000 = 3,528,000,000 > Int32.MaxValue
        //
        // そのため、乗算より前に double に昇格させます。
        double exactOutputLength =
            (double)source.Length * destinationRate / sourceRate;

        if (exactOutputLength > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Resample output is too large: {exactOutputLength:F0} samples"
            );
        }

        int outputLength = Math.Max(
            1,
            (int)Math.Round(exactOutputLength)
        );

        return ResampleToLength(source, outputLength);
    }

    private static float[] ResampleToLength(float[] source, int outputLength)
    {
        if (outputLength <= 0)
            return Array.Empty<float>();

        if (source.Length == 0)
            return new float[outputLength];

        if (source.Length == outputLength)
            return (float[])source.Clone();

        if (outputLength == 1)
            return new[] { source[0] };

        float[] output = new float[outputLength];
        float scale = (source.Length - 1) / (float)(outputLength - 1);

        for (int i = 0; i < outputLength; i++)
        {
            float position = i * scale;
            int left = Mathf.FloorToInt(position);
            int right = Mathf.Min(left + 1, source.Length - 1);
            float frac = position - left;
            output[i] = Mathf.Lerp(source[left], source[right], frac);
        }

        return output;
    }

    private static float[] UpsampleRvcV2FeaturesToFixed(
        float[] source,
        int sourceFrames,
        int destinationFrames,
        int channels
    )
    {
        if (sourceFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceFrames));

        float[] output =
            new float[destinationFrames * channels];

        int doubledFrames =
            Mathf.Min(sourceFrames * 2, destinationFrames);

        for (int t = 0; t < doubledFrames; t++)
        {
            int srcT = t / 2;

            Array.Copy(
                source,
                srcT * channels,
                output,
                t * channels,
                channels
            );
        }

        // Static 200-frame generator compatibility:
        // repeat the last real feature for any residual frames (normally 2).
        int lastSourceFrame = sourceFrames - 1;

        for (int t = doubledFrames; t < destinationFrames; t++)
        {
            Array.Copy(
                source,
                lastSourceFrame * channels,
                output,
                t * channels,
                channels
            );
        }

        return output;
    }

    private static void ApplyProtect(
        float[] processed,
        float[] original,
        float[] pitchf,
        float protectValue
    )
    {
        if (processed == null || original == null || pitchf == null)
            return;

        if (protectValue >= 0.5f)
            return;

        int frames = Mathf.Min(
            pitchf.Length,
            processed.Length / PhoneChannels
        );

        for (int t = 0; t < frames; t++)
        {
            float featureWeight =
                pitchf[t] > 0.0f
                    ? 1.0f
                    : protectValue;

            float originalWeight =
                1.0f - featureWeight;

            int offset = t * PhoneChannels;

            for (int c = 0; c < PhoneChannels; c++)
            {
                int i = offset + c;

                processed[i] =
                    processed[i] * featureWeight +
                    original[i] * originalWeight;
            }
        }
    }

    private static float NextGaussian(System.Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return (float)(
            Math.Sqrt(-2.0 * Math.Log(u1)) *
            Math.Cos(2.0 * Math.PI * u2)
        );
    }

    private static void ValidateAudio(float[] audio)
    {
        if (audio == null || audio.Length == 0)
            throw new InvalidOperationException("出力音声が空です。");

        for (int i = 0; i < audio.Length; i++)
        {
            if (float.IsNaN(audio[i]) || float.IsInfinity(audio[i]))
                throw new InvalidOperationException($"出力音声が不正です。index={i}");
        }
    }

    private static List<int> BuildOverlappingChunkStarts(
        int totalSamples,
        int chunkSamples,
        int strideSamples
    )
    {
        var starts = new List<int>();

        if (totalSamples <= chunkSamples)
        {
            starts.Add(0);
            return starts;
        }

        int start = 0;

        while (start + chunkSamples < totalSamples)
        {
            starts.Add(start);
            start += strideSamples;
        }

        int lastStart =
            Mathf.Max(0, totalSamples - chunkSamples);

        if (starts.Count == 0 ||
            starts[starts.Count - 1] != lastStart)
        {
            starts.Add(lastStart);
        }

        return starts;
    }

    private static void AddChunkOverlap(
        float[] sum,
        float[] weightSum,
        float[] chunk,
        int outputStart,
        int fadeSamples,
        bool isFirst,
        bool isLast
    )
    {
        int usable = Mathf.Min(
            chunk.Length,
            sum.Length - outputStart
        );

        if (usable <= 0)
            return;

        int fade = Mathf.Clamp(
            fadeSamples,
            0,
            chunk.Length / 2
        );

        for (int i = 0; i < usable; i++)
        {
            float w = 1.0f;

            if (!isFirst && fade > 0 && i < fade)
            {
                float x =
                    (i + 0.5f) / fade;

                float s =
                    Mathf.Sin(0.5f * Mathf.PI * x);

                w *= s * s;
            }

            if (!isLast &&
                fade > 0 &&
                i >= chunk.Length - fade)
            {
                float x =
                    (i - (chunk.Length - fade) + 0.5f) / fade;

                float c =
                    Mathf.Cos(0.5f * Mathf.PI * x);

                w *= c * c;
            }

            int dst = outputStart + i;

            sum[dst] += chunk[i] * w;
            weightSum[dst] += w;
        }
    }

    private static float[] FinalizeOverlapAdd(
        float[] sum,
        float[] weightSum
    )
    {
        float[] output = new float[sum.Length];

        for (int i = 0; i < output.Length; i++)
        {
            if (weightSum[i] > 1e-8f)
                output[i] = sum[i] / weightSum[i];
            else
                output[i] = 0.0f;
        }

        return output;
    }

    private static void NormalizeInPlace(float[] audio, float targetPeak)
    {
        float peak = 0.0f;
        for (int i = 0; i < audio.Length; i++)
            peak = Mathf.Max(peak, Mathf.Abs(audio[i]));

        if (peak < 1e-8f)
            return;

        float gain = targetPeak / peak;
        for (int i = 0; i < audio.Length; i++)
            audio[i] = Mathf.Clamp(audio[i] * gain, -1.0f, 1.0f);
    }

    private static void WriteMonoPcm16Wav(string path, float[] samples, int sampleRate)
    {
        if (samples == null)
            throw new ArgumentNullException(nameof(samples));

        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));

        string directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        const short channels = 1;
        const short bitsPerSample = 16;
        const short pcmFormat = 1;

        const int wavHeaderSize = 44;
        int bytesPerSample = bitsPerSample / 8;

        long dataSizeLong = (long)samples.Length * channels * bytesPerSample;
        if (dataSizeLong > int.MaxValue - 36)
        {
            throw new InvalidOperationException(
                $"WAVが大きすぎます。dataSize={dataSizeLong}"
            );
        }

        int dataSize = (int)dataSizeLong;
        int byteRate = sampleRate * channels * bytesPerSample;
        short blockAlign = (short)(channels * bytesPerSample);

        using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
        using (var writer = new BinaryWriter(stream))
        {
            // FourCCはchar[]ではなく、必ず1文字=1byteのASCIIとして書く。
            WriteFourCC(writer, "RIFF");
            writer.Write(36 + dataSize);
            WriteFourCC(writer, "WAVE");

            WriteFourCC(writer, "fmt ");
            writer.Write(16);                // PCM fmt chunk size
            writer.Write(pcmFormat);         // PCM = 1
            writer.Write(channels);          // mono
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write(blockAlign);
            writer.Write(bitsPerSample);

            WriteFourCC(writer, "data");
            writer.Write(dataSize);

            for (int i = 0; i < samples.Length; i++)
            {
                float s = Mathf.Clamp(samples[i], -1.0f, 1.0f);

                // -1.0 も -32768 まで使う。
                int pcmValue = s >= 0.0f
                    ? Mathf.RoundToInt(s * 32767.0f)
                    : Mathf.RoundToInt(s * 32768.0f);

                pcmValue = Mathf.Clamp(pcmValue, short.MinValue, short.MaxValue);
                writer.Write((short)pcmValue);
            }

            writer.Flush();
            stream.Flush(true);

            long expectedLength = wavHeaderSize + dataSize;
            if (stream.Length != expectedLength)
            {
                throw new InvalidOperationException(
                    $"WAVサイズ不一致: actual={stream.Length}, expected={expectedLength}"
                );
            }
        }
    }

    private static void WriteFourCC(BinaryWriter writer, string text)
    {
        if (text == null || text.Length != 4)
            throw new ArgumentException("FourCC must be exactly 4 ASCII characters.");

        for (int i = 0; i < 4; i++)
        {
            char c = text[i];
            if (c > 0x7F)
                throw new ArgumentException($"FourCC contains non-ASCII character: {text}");

            writer.Write((byte)c);
        }
    }

    private static void AnalyzeSignal(float[] audio, out float peak, out float rms)
    {
        peak = 0.0f;
        double sumSquares = 0.0;

        for (int i = 0; i < audio.Length; i++)
        {
            float s = audio[i];
            peak = Mathf.Max(peak, Mathf.Abs(s));
            sumSquares += (double)s * s;
        }

        rms = audio.Length > 0
            ? Mathf.Sqrt((float)(sumSquares / audio.Length))
            : 0.0f;
    }

    private static void ValidateWrittenWav(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists)
            throw new FileNotFoundException("保存したWAVが見つかりません。", path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var reader = new BinaryReader(stream);

        if (stream.Length < 44)
            throw new InvalidDataException($"WAVが44byte未満です: {stream.Length} bytes");

        string riff = ReadFourCC(reader);
        int riffSize = reader.ReadInt32();
        string wave = ReadFourCC(reader);
        string fmt = ReadFourCC(reader);
        int fmtSize = reader.ReadInt32();
        short format = reader.ReadInt16();
        short channels = reader.ReadInt16();
        int sampleRate = reader.ReadInt32();
        int byteRate = reader.ReadInt32();
        short blockAlign = reader.ReadInt16();
        short bitsPerSample = reader.ReadInt16();
        string data = ReadFourCC(reader);
        int dataSize = reader.ReadInt32();

        if (riff != "RIFF" || wave != "WAVE" || fmt != "fmt " || data != "data")
        {
            throw new InvalidDataException(
                $"WAVヘッダが不正です: RIFF={riff}, WAVE={wave}, fmt={fmt}, data={data}"
            );
        }

        if (format != 1 || channels != 1 || bitsPerSample != 16)
        {
            throw new InvalidDataException(
                $"想定外のWAV形式です: format={format}, channels={channels}, bits={bitsPerSample}"
            );
        }

        int bytesPerSample = channels * (bitsPerSample / 8);
        long sampleCount = bytesPerSample > 0 ? dataSize / bytesPerSample : 0;
        double duration = sampleRate > 0 ? sampleCount / (double)sampleRate : 0.0;

        Debug.Log(
            $"[RVC] WAV verify: fileBytes={info.Length}, riffSize={riffSize}, " +
            $"fmtSize={fmtSize}, sampleRate={sampleRate}, byteRate={byteRate}, " +
            $"blockAlign={blockAlign}, dataBytes={dataSize}, samples={sampleCount}, " +
            $"duration={duration:F4}s"
        );
    }

    private static string ReadFourCC(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);
        if (bytes.Length != 4)
            throw new EndOfStreamException("FourCCを4byte読み取れませんでした.");

        return new string(new[]
        {
            (char)bytes[0],
            (char)bytes[1],
            (char)bytes[2],
            (char)bytes[3]
        });
    }

    private static void LogModelIO(string label, Model model)
    {
        Debug.Log($"[RVC] ---- {label} model I/O ----");
        foreach (var input in model.inputs)
            Debug.Log($"[RVC] {label} input: {input.name} shape={input.shape} type={input.dataType}");
        foreach (var output in model.outputs)
            Debug.Log($"[RVC] {label} output: {output.name}");
    }

    private void OnDestroy()
    {
        contentWorker?.Dispose();
        rvcWorker?.Dispose();
        contentWorker = null;
        rvcWorker = null;
    }
}