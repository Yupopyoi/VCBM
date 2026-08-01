using System.IO;
using UnityEngine;

namespace VCBM.VoiceClone
{
    [RequireComponent(typeof(VoiceCloneServiceClient))]
    public sealed class VoiceClonePrototypePanel :
        MonoBehaviour
    {
        [SerializeField]
        private bool showPanel = true;

        [SerializeField]
        private float panelWidth = 820f;

        [Header("Reference audio")]
        [SerializeField]
        private string referencePath = "";

        [Header("Clone speech")]
        [SerializeField, TextArea(2, 5)]
        private string speechText =
            "こんにちは。今日は少し早く起きたので、近くまで散歩してきました。この声は、ちゃんと自然に聞こえていますか？";

        [SerializeField, TextArea(2, 5)]
        private string stylePrompt =
            "Cheerful and energetic, with natural lively intonation and an audible smile.";

        [Header("Output mode")]
        [SerializeField]
        private bool fiveMinuteMode = false;

        [Header("Generation settings")]
        [SerializeField, Range(1f, 3f)]
        private float cfgValue = 1.5f;

        [SerializeField, Range(4, 30)]
        private int inferenceTimesteps = 20;

        [SerializeField]
        private int seed = 42;

        [SerializeField, Range(1, 10)]
        private int candidateCount = 1;

        [SerializeField]
        private bool normalize = true;

        private VoiceCloneServiceClient serviceClient;
        private Vector2 scrollPosition;
        private string localStatus = "Ready";

        private void Awake()
        {
            serviceClient =
                GetComponent<VoiceCloneServiceClient>();
        }

        private void OnGUI()
        {
            if (!showPanel || serviceClient == null)
            {
                return;
            }

            float width = Mathf.Min(
                panelWidth,
                Screen.width - 40f);

            float height = Mathf.Max(
                200f,
                Screen.height - 40f);

            GUILayout.BeginArea(
                new Rect(
                    1920/2f,
                    20f,
                    width,
                    height),
                GUI.skin.window);

            GUILayout.Label(
                "VCBM Voice Clone Prototype");

            scrollPosition =
                GUILayout.BeginScrollView(
                    scrollPosition);

            DrawReferenceSection();
            DrawCloneSettingsSection();
            DrawCandidateSection();

            GUILayout.Space(12f);

            GUILayout.Label(
                "Local status: " + localStatus);

            GUILayout.Label(
                "Service status: " +
                serviceClient.Status);

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private void DrawReferenceSection()
        {
            GUILayout.Label("Reference audio");

            GUILayout.BeginHorizontal();

            referencePath = GUILayout.TextField(
                referencePath ?? string.Empty,
                GUILayout.MinWidth(560f),
                GUILayout.Height(30f));

            bool oldEnabled = GUI.enabled;
            GUI.enabled = !serviceClient.IsBusy;

            if (GUILayout.Button(
                    "Select WAV",
                    GUILayout.Width(130f),
                    GUILayout.Height(30f)))
            {
                string selectedPath =
                    LocalWavFilePicker.Open();

                if (!string.IsNullOrWhiteSpace(
                        selectedPath))
                {
                    referencePath = selectedPath;

                    localStatus =
                        "Selected: " +
                        Path.GetFileName(
                            selectedPath);
                }
            }

            GUI.enabled = oldEnabled;

            GUILayout.EndHorizontal();

            if (!string.IsNullOrWhiteSpace(
                    referencePath))
            {
                GUILayout.Label(
                    "File: " +
                    Path.GetFileName(
                        referencePath));
            }
        }

        private void DrawCloneSettingsSection()
        {
            GUILayout.Space(12f);

            fiveMinuteMode = GUILayout.Toggle(
                fiveMinuteMode,
                "5-minute output mode");

            if (fiveMinuteMode)
            {
                GUILayout.Label(
                    "Python側の定型文章を使って、" +
                    "合計300秒以上になるまで生成します。");

                GUILayout.Label(
                    "このモードでは下のSpeech textは使用されません。");
            }

            GUILayout.Space(8f);
            GUILayout.Label("Speech text");

            bool previousEnabled = GUI.enabled;

            // 5分モード中も表示は残すが編集不可
            GUI.enabled =
                !fiveMinuteMode &&
                !serviceClient.IsBusy;

            speechText = GUILayout.TextArea(
                speechText ?? string.Empty,
                GUILayout.MinHeight(80f));

            GUI.enabled = previousEnabled;

            GUILayout.Space(8f);
            GUILayout.Label("Style instruction");

            stylePrompt = GUILayout.TextArea(
                stylePrompt ?? string.Empty,
                GUILayout.MinHeight(70f));

            cfgValue = DrawFloatField(
                "CFG (1.0 - 3.0)",
                cfgValue,
                1f,
                3f);

            inferenceTimesteps = DrawIntField(
                "Steps (4 - 30)",
                inferenceTimesteps,
                4,
                30);

            seed = DrawIntField(
                "Seed",
                seed,
                int.MinValue,
                int.MaxValue);

            candidateCount = DrawIntField(
                "Candidates (1 - 10)",
                candidateCount,
                1,
                10);

            normalize = GUILayout.Toggle(
                normalize,
                "Normalize");

            GUILayout.Space(8f);
            GUILayout.BeginHorizontal();

            bool oldEnabled = GUI.enabled;
            GUI.enabled = !serviceClient.IsBusy;

            if (GUILayout.Button(
                    "Check Service",
                    GUILayout.Height(36f)))
            {
                serviceClient.CheckHealth();
                localStatus =
                    "Service check requested";
            }

            if (GUILayout.Button(
                    "Clone Voice",
                    GUILayout.Height(36f)))
            {
                serviceClient.Clone(
                    referencePath,
                    speechText,
                    stylePrompt,
                    fiveMinuteMode,
                    cfgValue,
                    inferenceTimesteps,
                    seed,
                    candidateCount,
                    normalize);

                localStatus =
                    "Clone requested";
            }

            if (GUILayout.Button(
                    "Stop",
                    GUILayout.Width(100f),
                    GUILayout.Height(36f)))
            {
                serviceClient.StopPlayback();
                localStatus =
                    "Playback stopped";
            }

            GUI.enabled = oldEnabled;

            GUILayout.EndHorizontal();
        }

        private void DrawCandidateSection()
        {
            VoiceCloneResponse response =
                serviceClient.LastResponse;

            if (response == null ||
                response.candidates == null)
            {
                return;
            }

            GUILayout.Space(12f);
            GUILayout.Label("Generated candidates");

            for (int index = 0;
                 index < response.candidates.Length;
                 index++)
            {
                VoiceCloneCandidate candidate =
                    response.candidates[index];

                GUILayout.BeginHorizontal();

                GUILayout.Label(
                    candidate.file_name +
                    " / seed " +
                    candidate.seed +
                    " / " +
                    candidate.duration_seconds
                        .ToString("0.00") +
                    " sec" +
                    " / generation " +
                    candidate.generation_seconds
                        .ToString("0.00") +
                    " sec",
                    GUILayout.Width(620f));

                bool oldEnabled = GUI.enabled;
                GUI.enabled = !serviceClient.IsBusy;

                int selectedIndex = index;

                if (GUILayout.Button(
                        "Play",
                        GUILayout.Width(100f)))
                {
                    serviceClient.PlayCandidate(
                        selectedIndex);
                }

                GUI.enabled = oldEnabled;

                GUILayout.EndHorizontal();
            }
        }

        private static float DrawFloatField(
            string label,
            float value,
            float minimum,
            float maximum)
        {
            GUILayout.BeginHorizontal();

            GUILayout.Label(
                label,
                GUILayout.Width(170f));

            string text = GUILayout.TextField(
                value.ToString("0.00"),
                GUILayout.Width(110f));

            GUILayout.EndHorizontal();

            if (float.TryParse(
                    text,
                    out float parsed))
            {
                return Mathf.Clamp(
                    parsed,
                    minimum,
                    maximum);
            }

            return value;
        }

        private static int DrawIntField(
            string label,
            int value,
            int minimum,
            int maximum)
        {
            GUILayout.BeginHorizontal();

            GUILayout.Label(
                label,
                GUILayout.Width(170f));

            string text = GUILayout.TextField(
                value.ToString(),
                GUILayout.Width(110f));

            GUILayout.EndHorizontal();

            if (!int.TryParse(
                    text,
                    out int parsed))
            {
                return value;
            }

            if (parsed < minimum)
            {
                return minimum;
            }

            if (parsed > maximum)
            {
                return maximum;
            }

            return parsed;
        }
    }
}
