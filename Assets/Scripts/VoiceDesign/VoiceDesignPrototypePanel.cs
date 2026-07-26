using UnityEngine;

namespace VCBM.VoiceDesign
{
    [RequireComponent(typeof(VoiceDesignController))]
    public sealed class VoiceDesignPrototypePanel : MonoBehaviour
    {
        [SerializeField] private bool showPanel = true;
        [SerializeField] private float panelWidth = 760f;

        private VoiceDesignController controller;
        private Vector2 scrollPosition;
        private string status = "Ready";

        private void Awake()
        {
            controller = GetComponent<VoiceDesignController>();
        }

        private void OnGUI()
        {
            if (!showPanel || controller == null)
            {
                return;
            }

            float width = Mathf.Min(panelWidth, Screen.width - 40f);
            float height = Mathf.Max(200f, Screen.height - 40f);

            GUILayout.BeginArea(
                new Rect(20f, 20f, width, height),
                GUI.skin.window);

            GUILayout.Label("VCBM Voice Design Prototype");
            GUILayout.Label("Unity slider settings -> VoxCPM2 voice prompt");

            scrollPosition = GUILayout.BeginScrollView(scrollPosition);

            VoiceStyleSettings s = controller.Settings;
            bool changed = false;

            changed |= DrawSlider("Youthful", ref s.youthful);
            changed |= DrawSlider("Pitch", ref s.pitch);
            changed |= DrawSlider("Brightness", ref s.brightness);
            changed |= DrawSlider("Softness", ref s.softness);
            changed |= DrawSlider("Breathiness", ref s.breathiness);
            changed |= DrawSlider("Energy", ref s.energy);
            changed |= DrawSlider("Cuteness", ref s.cuteness);
            changed |= DrawSlider("Expressiveness", ref s.expressiveness);
            changed |= DrawSlider("Speed", ref s.speed);
            changed |= DrawSlider("Naturalness", ref s.naturalness);

            GUILayout.Space(8f);
            GUILayout.Label("Additional instruction");

            string custom = GUILayout.TextArea(
                s.customInstruction ?? string.Empty,
                GUILayout.MinHeight(60f));

            if (custom != s.customInstruction)
            {
                s.customInstruction = custom;
                changed = true;
            }

            if (changed)
            {
                controller.RebuildPrompt();
                status = "Prompt updated";
            }

            GUILayout.Space(12f);
            GUILayout.BeginHorizontal();

            if (GUILayout.Button("Reset: Natural Charming", GUILayout.Height(32f)))
            {
                controller.ApplyNaturalCharmingPreset();
                status = "Preset restored";
            }

            if (GUILayout.Button("Copy Prompt", GUILayout.Height(32f)))
            {
                controller.CopyPromptToClipboard();
                status = "Copied to clipboard";
            }

            GUILayout.EndHorizontal();

            GUILayout.Space(12f);
            GUILayout.Label("Generated VoxCPM2 prompt");

            bool oldEnabled = GUI.enabled;
            GUI.enabled = false;
            GUILayout.TextArea(
                controller.CurrentPrompt,
                GUILayout.MinHeight(180f));
            GUI.enabled = oldEnabled;

            GUILayout.Space(8f);
            GUILayout.Label("Status: " + status);

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private static bool DrawSlider(string label, ref float value)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(120f));

            float newValue = GUILayout.HorizontalSlider(
                value,
                0f,
                1f,
                GUILayout.Width(480f));

            GUILayout.Label(newValue.ToString("0.00"), GUILayout.Width(48f));
            GUILayout.EndHorizontal();

            if (Mathf.Approximately(newValue, value))
            {
                return false;
            }

            value = newValue;
            return true;
        }
    }
}
