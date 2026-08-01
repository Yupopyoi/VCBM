using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace VCBM.VoxCPM
{
    [DisallowMultipleComponent]
    public sealed class VoxCpmServiceClient : MonoBehaviour
    {
        [Header("Local VoxCPM2 service")]
        [SerializeField] private string baseUrl = "http://127.0.0.1:8765";

        [Header("Audio playback")]
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private bool autoPlayFirstCandidate = true;

        public bool IsBusy { get; private set; }
        public string Status { get; private set; } = "Not checked";
        public VoxCpmGenerateResponse LastResponse { get; private set; }

        public event Action<string> StatusChanged;
        public event Action<VoxCpmGenerateResponse> GenerationCompleted;
        public event Action<string> RequestFailed;

        private void Awake()
        {
            EnsureAudioSource();
        }

        public void CheckHealth()
        {
            if (!IsBusy)
            {
                StartCoroutine(CheckHealthCoroutine());
            }
        }

        public void Generate(
            string voicePrompt,
            string speechText,
            float cfgValue,
            int inferenceTimesteps,
            int seed,
            int candidateCount,
            bool normalize)
        {
            if (IsBusy)
            {
                SetStatus("Another request is running.");
                return;
            }

            if (string.IsNullOrWhiteSpace(voicePrompt))
            {
                Fail("Voice prompt is empty.");
                return;
            }

            if (string.IsNullOrWhiteSpace(speechText))
            {
                Fail("Speech text is empty.");
                return;
            }

            var requestBody = new VoxCpmGenerateRequest
            {
                voice_prompt = voicePrompt.Trim(),
                speech_text = speechText.Trim(),
                cfg_value = Mathf.Clamp(cfgValue, 1f, 3f),
                inference_timesteps = Mathf.Clamp(inferenceTimesteps, 4, 30),
                seed = seed,
                candidate_count = Mathf.Clamp(candidateCount, 1, 10),
                normalize = normalize
            };

            StartCoroutine(GenerateCoroutine(requestBody));
        }

        public void PlayCandidate(int index)
        {
            if (LastResponse == null ||
                LastResponse.candidates == null ||
                index < 0 ||
                index >= LastResponse.candidates.Length)
            {
                Fail("Candidate index is invalid.");
                return;
            }

            if (!IsBusy)
            {
                StartCoroutine(DownloadAndPlayCoroutine(
                    LastResponse.candidates[index]));
            }
        }

        private IEnumerator CheckHealthCoroutine()
        {
            IsBusy = true;
            SetStatus("Checking VoxCPM2 service...");

            string url = CombineUrl("/health");

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = 10;
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    IsBusy = false;
                    Fail("Service check failed: " + request.error);
                    yield break;
                }

                VoxCpmHealthResponse response;
                try
                {
                    response = JsonUtility.FromJson<VoxCpmHealthResponse>(
                        request.downloadHandler.text);
                }
                catch (Exception exception)
                {
                    IsBusy = false;
                    Fail("Invalid health response: " + exception.Message);
                    yield break;
                }

                IsBusy = false;
                SetStatus(
                    "Service OK / model: " + response.model_state +
                    " / device: " + response.device);
            }
        }

        private IEnumerator GenerateCoroutine(VoxCpmGenerateRequest body)
        {
            IsBusy = true;
            LastResponse = null;
            SetStatus("Generating voice. The first request also loads VoxCPM2...");

            string json = JsonUtility.ToJson(body);
            byte[] payload = Encoding.UTF8.GetBytes(json);
            string url = CombineUrl("/voice/generate");

            using (UnityWebRequest request = new UnityWebRequest(
                url,
                UnityWebRequest.kHttpVerbPOST))
            {
                request.uploadHandler = new UploadHandlerRaw(payload);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.timeout = 900;

                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    IsBusy = false;
                    Fail(
                        "Generation failed: " + request.error +
                        "\n" + request.downloadHandler.text);
                    yield break;
                }

                VoxCpmGenerateResponse response;
                try
                {
                    response = JsonUtility.FromJson<VoxCpmGenerateResponse>(
                        request.downloadHandler.text);
                }
                catch (Exception exception)
                {
                    IsBusy = false;
                    Fail("Invalid generation response: " + exception.Message);
                    yield break;
                }

                if (response == null || !response.success)
                {
                    IsBusy = false;
                    Fail(response != null ? response.error : "Empty response.");
                    yield break;
                }

                LastResponse = response;
                IsBusy = false;

                int count = response.candidates != null
                    ? response.candidates.Length
                    : 0;

                SetStatus("Generated " + count + " candidate(s).");
                GenerationCompleted?.Invoke(response);

                if (autoPlayFirstCandidate && count > 0)
                {
                    yield return DownloadAndPlayCoroutine(
                        response.candidates[0]);
                }
            }
        }

        private IEnumerator DownloadAndPlayCoroutine(VoxCpmCandidate candidate)
        {
            if (candidate == null || string.IsNullOrWhiteSpace(candidate.audio_url))
            {
                Fail("Candidate audio URL is empty.");
                yield break;
            }

            IsBusy = true;
            SetStatus("Downloading " + candidate.file_name + "...");

            string url = candidate.audio_url.StartsWith(
                "http",
                StringComparison.OrdinalIgnoreCase)
                ? candidate.audio_url
                : CombineUrl(candidate.audio_url);

            using (UnityWebRequest request =
                UnityWebRequestMultimedia.GetAudioClip(url, AudioType.WAV))
            {
                request.timeout = 120;
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    IsBusy = false;
                    Fail("Audio download failed: " + request.error);
                    yield break;
                }

                AudioClip clip = DownloadHandlerAudioClip.GetContent(request);
                if (clip == null)
                {
                    IsBusy = false;
                    Fail("Downloaded audio could not be decoded.");
                    yield break;
                }

                EnsureAudioSource();
                audioSource.Stop();
                audioSource.clip = clip;
                audioSource.Play();

                IsBusy = false;
                SetStatus("Playing " + candidate.file_name);
            }
        }

        private string CombineUrl(string path)
        {
            return baseUrl.TrimEnd('/') + "/" + path.TrimStart('/');
        }

        private void EnsureAudioSource()
        {
            if (audioSource == null)
            {
                audioSource = GetComponent<AudioSource>();
            }

            if (audioSource == null)
            {
                audioSource = gameObject.AddComponent<AudioSource>();
                audioSource.playOnAwake = false;
            }
        }

        private void SetStatus(string message)
        {
            Status = message;
            Debug.Log("[VoxCPM2] " + message);
            StatusChanged?.Invoke(message);
        }

        private void Fail(string message)
        {
            Status = message;
            Debug.LogError("[VoxCPM2] " + message);
            StatusChanged?.Invoke(message);
            RequestFailed?.Invoke(message);
        }
    }
}
