using UnityEngine;
using VCBM.VoxCPM;

namespace VCBM.VoiceDesign
{
    /// <summary>
    /// 声質スライダー、VoxCPM2生成、候補再生を一画面で確認する仮UI。
    /// 正式版ではCanvasまたはUI Toolkitへ置き換える。
    /// </summary>
    [RequireComponent(typeof(VoiceDesignController))]
    [RequireComponent(typeof(VoxCpmServiceClient))]
    public sealed class VoiceDesignPrototypePanel : MonoBehaviour
    {
        [SerializeField] private bool showPanel = true;
        [SerializeField] private float panelWidth = 820f;

        [Header("VoxCPM2 test generation")]
        [SerializeField, TextArea(2, 5)]
        private string speechText =
            "こんにちは。今日は少し早く起きたので、近くまで散歩してきました。" +
            "この声は、ちゃんと自然に聞こえていますか？";

        [SerializeField, Range(1f, 3f)] private float cfgValue = 1.5f;
        [SerializeField, Range(4, 30)] private int inferenceTimesteps = 20;
        [SerializeField] private int seed = 42;
        [SerializeField, Range(1, 10)] private int candidateCount = 1;
        [SerializeField] private bool normalizeText = true;

        private VoiceDesignController controller;
        private VoxCpmServiceClient serviceClient;
        private Vector2 scrollPosition;
        private string localStatus = "Ready";

        private void Awake()
        {
            controller = GetComponent<VoiceDesignController>();
            serviceClient = GetComponent<VoxCpmServiceClient>();
        }

        private void OnGUI()
        {
            if (!showPanel || controller == null || serviceClient == null)
            {
                return;
            }

            float width = Mathf.Min(panelWidth, Screen.width - 40f);
            float height = Mathf.Max(200f, Screen.height - 40f);

            GUILayout.BeginArea(
                new Rect(20f, 20f, width, height),
                GUI.skin.window);

            GUILayout.Label("VCBM Voice Design + VoxCPM2 Prototype");

            scrollPosition = GUILayout.BeginScrollView(scrollPosition);

            DrawStyleSection();
            DrawPromptSection();
            DrawGenerationSection();
            DrawCandidateSection();

            GUILayout.Space(10f);
            GUILayout.Label("Local status: " + localStatus);
            GUILayout.Label("Service status: " + serviceClient.Status);

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private void DrawStyleSection()
        {
            GUILayout.Label("Voice style");

            VoiceStyleSettings settings = controller.Settings;
            bool changed = false;

            changed |= DrawSlider("Youthful", ref settings.youthful);
            changed |= DrawSlider("Pitch", ref settings.pitch);
            changed |= DrawSlider("Brightness", ref settings.brightness);
            changed |= DrawSlider("Softness", ref settings.softness);
            changed |= DrawSlider("Breathiness", ref settings.breathiness);
            changed |= DrawSlider("Energy", ref settings.energy);
            changed |= DrawSlider("Cuteness", ref settings.cuteness);
            changed |= DrawSlider(
                "Expressiveness",
                ref settings.expressiveness);
            changed |= DrawSlider("Speed", ref settings.speed);
            changed |= DrawSlider("Naturalness", ref settings.naturalness);

            GUILayout.Space(6f);
            GUILayout.Label("Additional English instruction");

            string custom = GUILayout.TextArea(
                settings.customInstruction ?? string.Empty,
                GUILayout.MinHeight(54f));

            if (custom != settings.customInstruction)
            {
                settings.customInstruction = custom;
                changed = true;
            }

            if (changed)
            {
                controller.RebuildPrompt();
                localStatus = "Prompt updated";
            }

            GUILayout.BeginHorizontal();

            if (GUILayout.Button(
                "Reset: Natural Charming",
                GUILayout.Height(30f)))
            {
                controller.ApplyNaturalCharmingPreset();
                localStatus = "Preset restored";
            }

            if (GUILayout.Button("Copy Prompt", GUILayout.Height(30f)))
            {
                controller.CopyPromptToClipboard();
                localStatus = "Prompt copied";
            }

            GUILayout.EndHorizontal();
        }

        private void DrawPromptSection()
        {
            GUILayout.Space(12f);
            GUILayout.Label("Generated VoxCPM2 prompt");

            bool oldEnabled = GUI.enabled;
            GUI.enabled = false;
            GUILayout.TextArea(
                controller.CurrentPrompt,
                GUILayout.MinHeight(150f));
            GUI.enabled = oldEnabled;
        }

        private void DrawGenerationSection()
        {
            GUILayout.Space(12f);
            GUILayout.Label("Test speech");

            speechText = GUILayout.TextArea(
                speechText,
                GUILayout.MinHeight(72f));

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

            normalizeText = GUILayout.Toggle(
                normalizeText,
                "Normalize text");

            GUILayout.BeginHorizontal();

            bool oldEnabled = GUI.enabled;
            GUI.enabled = !serviceClient.IsBusy;

            if (GUILayout.Button(
                "Check Service",
                GUILayout.Height(34f)))
            {
                serviceClient.CheckHealth();
            }

            if (GUILayout.Button(
                "Generate Voice",
                GUILayout.Height(34f)))
            {
                serviceClient.Generate(
                    controller.CurrentPrompt,
                    speechText,
                    cfgValue,
                    inferenceTimesteps,
                    seed,
                    candidateCount,
                    normalizeText);

                localStatus = "Generation requested";
            }

            GUI.enabled = oldEnabled;
            GUILayout.EndHorizontal();
        }

        private void DrawCandidateSection()
        {
            VoxCpmGenerateResponse response = serviceClient.LastResponse;

            if (response == null || response.candidates == null)
            {
                return;
            }

            GUILayout.Space(12f);
            GUILayout.Label("Generated candidates");

            for (int i = 0; i < response.candidates.Length; i++)
            {
                VoxCpmCandidate candidate = response.candidates[i];

                GUILayout.BeginHorizontal();
                GUILayout.Label(
                    candidate.file_name +
                    " / seed " + candidate.seed +
                    " / " + candidate.duration_seconds.ToString("0.00") + " sec",
                    GUILayout.Width(590f));

                bool oldEnabled = GUI.enabled;
                GUI.enabled = !serviceClient.IsBusy;

                if (GUILayout.Button("Play", GUILayout.Width(100f)))
                {
                    serviceClient.PlayCandidate(i);
                }

                GUI.enabled = oldEnabled;
                GUILayout.EndHorizontal();
            }
        }

        private static bool DrawSlider(string label, ref float value)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(130f));

            float newValue = GUILayout.HorizontalSlider(
                value,
                0f,
                1f,
                GUILayout.Width(540f));

            GUILayout.Label(newValue.ToString("0.00"), GUILayout.Width(52f));
            GUILayout.EndHorizontal();

            if (Mathf.Approximately(newValue, value))
            {
                return false;
            }

            value = newValue;
            return true;
        }

        private static float DrawFloatField(
            string label,
            float value,
            float minimum,
            float maximum)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(160f));
            string text = GUILayout.TextField(
                value.ToString("0.00"),
                GUILayout.Width(100f));
            GUILayout.EndHorizontal();

            float parsed;
            if (float.TryParse(text, out parsed))
            {
                return Mathf.Clamp(parsed, minimum, maximum);
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
            GUILayout.Label(label, GUILayout.Width(160f));
            string text = GUILayout.TextField(
                value.ToString(),
                GUILayout.Width(100f));
            GUILayout.EndHorizontal();

            int parsed;
            if (int.TryParse(text, out parsed))
            {
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

            return value;
        }
    }
}
