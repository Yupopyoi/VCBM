using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace VCBM.VoiceClone
{
    [RequireComponent(typeof(AudioSource))]
    public sealed class VoiceCloneServiceClient : MonoBehaviour
    {
        [SerializeField]
        private string serverUrl = "http://127.0.0.1:8765";

        public bool IsBusy { get; private set; }

        public string Status { get; private set; } = "Ready";

        public VoiceCloneResponse LastResponse { get; private set; }

        private AudioSource audioSource;
        private AudioClip loadedAudioClip;

        private void Awake()
        {
            audioSource = GetComponent<AudioSource>();

            audioSource.playOnAwake = false;
            audioSource.loop = false;
        }

        private void OnDestroy()
        {
            if (loadedAudioClip != null)
            {
                Destroy(loadedAudioClip);
            }
        }

        public void CheckHealth()
        {
            if (IsBusy)
            {
                return;
            }

            StartCoroutine(CheckHealthCoroutine());
        }

        public void Clone(
            string referencePath,
            string speechText,
            string stylePrompt,
            bool fiveMinuteMode,
            float cfgValue,
            int inferenceTimesteps,
            int seed,
            int candidateCount,
            bool normalize)
        {
            if (IsBusy)
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(referencePath))
            {
                Status = "参照音声を選択してください。";
                return;
            }

            if (!File.Exists(referencePath))
            {
                Status = "参照音声が見つかりません。";
                return;
            }

            if (!string.Equals(
                    Path.GetExtension(referencePath),
                    ".wav",
                    StringComparison.OrdinalIgnoreCase))
            {
                Status = "参照音声にはWAVを指定してください。";
                return;
            }

            if (string.IsNullOrWhiteSpace(speechText))
            {
                Status = "話させる文章を入力してください。";
                return;
            }

            StartCoroutine(
                CloneSequence(
                    referencePath,
                    speechText,
                    stylePrompt,
                    fiveMinuteMode,
                    cfgValue,
                    inferenceTimesteps,
                    seed,
                    candidateCount,
                    normalize));
        }

        public void PlayCandidate(int candidateIndex)
        {
            if (IsBusy)
            {
                return;
            }

            if (LastResponse == null ||
                LastResponse.candidates == null ||
                candidateIndex < 0 ||
                candidateIndex >= LastResponse.candidates.Length)
            {
                Status = "再生する候補がありません。";
                return;
            }

            StartCoroutine(
                PlayCandidateCoroutine(
                    LastResponse.candidates[candidateIndex]));
        }

        public void StopPlayback()
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
                Status = "再生を停止しました。";
            }
        }

        private IEnumerator CheckHealthCoroutine()
        {
            IsBusy = true;
            Status = "サービスを確認しています…";

            string url =
                $"{serverUrl.TrimEnd('/')}/health";

            using UnityWebRequest request =
                UnityWebRequest.Get(url);

            yield return request.SendWebRequest();

            if (request.result ==
                UnityWebRequest.Result.Success)
            {
                Status = "VoxCPM2サービスに接続できました。";
            }
            else
            {
                Status = BuildHttpError(
                    "サービスへの接続に失敗しました",
                    request);
            }

            IsBusy = false;
        }

        private IEnumerator CloneSequence(
            string referencePath,
            string speechText,
            string stylePrompt,
            bool fiveMinuteMode,
            float cfgValue,
            int inferenceTimesteps,
            int seed,
            int candidateCount,
            bool normalize)
        {
            IsBusy = true;
            LastResponse = null;

            Status = "参照音声を読み込んでいます…";

            byte[] wavBytes;

            try
            {
                wavBytes = File.ReadAllBytes(referencePath);
            }
            catch (Exception exception)
            {
                Status =
                    $"参照音声の読み込みに失敗しました: " +
                    exception.Message;

                IsBusy = false;
                yield break;
            }

            Status = "参照音声をアップロードしています…";

            var formSections =
                new List<IMultipartFormSection>
                {
                    new MultipartFormFileSection(
                        "file",
                        wavBytes,
                        Path.GetFileName(referencePath),
                        "audio/wav")
                };

            string uploadUrl =
                $"{serverUrl.TrimEnd('/')}" +
                "/voice/reference/upload";

            UploadReferenceResponse uploadResponse;

            using (UnityWebRequest uploadRequest =
                   UnityWebRequest.Post(
                       uploadUrl,
                       formSections))
            {
                yield return uploadRequest.SendWebRequest();

                if (uploadRequest.result !=
                    UnityWebRequest.Result.Success)
                {
                    Status = BuildHttpError(
                        "参照音声のアップロードに失敗しました",
                        uploadRequest);

                    IsBusy = false;
                    yield break;
                }

                try
                {
                    uploadResponse =
                        JsonUtility.FromJson<
                            UploadReferenceResponse>(
                            uploadRequest.downloadHandler.text);
                }
                catch (Exception exception)
                {
                    Status =
                        $"アップロード応答を解析できません: " +
                        exception.Message;

                    IsBusy = false;
                    yield break;
                }
            }

            if (uploadResponse == null ||
                !uploadResponse.success ||
                string.IsNullOrWhiteSpace(
                    uploadResponse.reference_file_name))
            {
                Status = "参照音声の登録に失敗しました。";
                IsBusy = false;
                yield break;
            }

            Status = "Voice Cloneを実行しています…";

            var cloneRequestBody =
                new VoiceCloneRequest
                {
                    reference_file_name =
                        uploadResponse.reference_file_name,

                    speech_text = speechText,
                    mode = "controllable",
                    style_prompt = stylePrompt,
                    reference_text = "",

                    five_minute_mode =
                        fiveMinuteMode,

                    cfg_value = cfgValue,
                    inference_timesteps =
                        inferenceTimesteps,
                    seed = seed,
                    candidate_count =
                        candidateCount,
                    normalize = normalize
                };

            string json =
                JsonUtility.ToJson(cloneRequestBody);

            byte[] requestBytes =
                Encoding.UTF8.GetBytes(json);

            string cloneUrl =
                $"{serverUrl.TrimEnd('/')}/voice/clone";

            using var cloneRequest =
                new UnityWebRequest(
                    cloneUrl,
                    UnityWebRequest.kHttpVerbPOST);

            cloneRequest.uploadHandler =
                new UploadHandlerRaw(requestBytes);

            cloneRequest.downloadHandler =
                new DownloadHandlerBuffer();

            cloneRequest.SetRequestHeader(
                "Content-Type",
                "application/json");

            yield return cloneRequest.SendWebRequest();

            if (cloneRequest.result !=
                UnityWebRequest.Result.Success)
            {
                Status = BuildHttpError(
                    "Voice Cloneに失敗しました",
                    cloneRequest);

                IsBusy = false;
                yield break;
            }

            try
            {
                LastResponse =
                    JsonUtility.FromJson<
                        VoiceCloneResponse>(
                        cloneRequest.downloadHandler.text);
            }
            catch (Exception exception)
            {
                Status =
                    $"Clone応答を解析できません: " +
                    exception.Message;

                IsBusy = false;
                yield break;
            }

            if (LastResponse == null ||
                !LastResponse.success ||
                LastResponse.candidates == null ||
                LastResponse.candidates.Length == 0)
            {
                Status = "Clone音声が生成されませんでした。";
                IsBusy = false;
                yield break;
            }

            if (LastResponse.five_minute_mode)
            {
                Status =
                    $"5分出力が完了しました。 " +
                    $"フォルダ: {LastResponse.folder_name} / " +
                    $"合計: {LastResponse.total_duration_seconds:0.00}秒 / " +
                    $"{LastResponse.candidates.Length}ファイル";
            }
            else
            {
                Status =
                    $"{LastResponse.candidates.Length}件の" +
                    "Clone音声を生成しました。";
            }

            IsBusy = false;
        }

        private IEnumerator PlayCandidateCoroutine(
            VoiceCloneCandidate candidate)
        {
            IsBusy = true;

            string audioUrl =
                ResolveServerUrl(candidate.audio_url);

            Status =
                $"音声を読み込んでいます: " +
                candidate.file_name;

            using UnityWebRequest request =
                UnityWebRequestMultimedia.GetAudioClip(
                    audioUrl,
                    AudioType.WAV);

            yield return request.SendWebRequest();

            if (request.result !=
                UnityWebRequest.Result.Success)
            {
                Status = BuildHttpError(
                    "Clone音声の取得に失敗しました",
                    request);

                IsBusy = false;
                yield break;
            }

            if (loadedAudioClip != null)
            {
                audioSource.Stop();
                Destroy(loadedAudioClip);
            }

            loadedAudioClip =
                DownloadHandlerAudioClip.GetContent(request);

            loadedAudioClip.name =
                candidate.file_name;

            audioSource.clip = loadedAudioClip;
            audioSource.Play();

            Status =
                $"再生中: {candidate.file_name}";

            IsBusy = false;
        }

        private string ResolveServerUrl(
            string audioUrl)
        {
            if (audioUrl.StartsWith(
                    "http://",
                    StringComparison.OrdinalIgnoreCase) ||
                audioUrl.StartsWith(
                    "https://",
                    StringComparison.OrdinalIgnoreCase))
            {
                return audioUrl;
            }

            return
                $"{serverUrl.TrimEnd('/')}/" +
                audioUrl.TrimStart('/');
        }

        private static string BuildHttpError(
            string heading,
            UnityWebRequest request)
        {
            string responseBody =
                request.downloadHandler?.text;

            if (string.IsNullOrWhiteSpace(
                    responseBody))
            {
                return
                    $"{heading}: {request.error}";
            }

            return
                $"{heading}: HTTP {request.responseCode}\n" +
                responseBody;
        }
    }

    [Serializable]
    public sealed class UploadReferenceResponse
    {
        public bool success;
        public string reference_file_name;
        public string original_file_name;
        public int sample_rate;
        public int channels;
        public float duration_seconds;
        public string error;
    }

    [Serializable]
    public sealed class VoiceCloneRequest
    {
        public string reference_file_name;
        public string speech_text;
        public string mode;
        public string style_prompt;
        public string reference_text;

        public bool five_minute_mode;

        public float cfg_value;
        public int inference_timesteps;
        public int seed;
        public int candidate_count;
        public bool normalize;
    }

    [Serializable]
    public sealed class VoiceCloneResponse
    {
        public bool success;
        public string request_id;
        public string mode;

        public bool five_minute_mode;
        public string folder_name;
        public float target_duration_seconds;
        public float total_duration_seconds;

        public string reference_file_name;
        public VoiceCloneCandidate[] candidates;
        public string error;
    }

    [Serializable]
    public sealed class VoiceCloneCandidate
    {
        public string candidate_id;
        public string audio_url;
        public string file_name;
        public int sample_rate;
        public int seed;
        public float duration_seconds;
        public float generation_seconds;
    }
}