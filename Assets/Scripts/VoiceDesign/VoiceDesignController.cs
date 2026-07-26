using System;
using UnityEngine;

namespace VCBM.VoiceDesign
{
    public sealed class VoiceDesignController : MonoBehaviour
    {
        [SerializeField]
        private VoiceStyleSettings settings =
            VoiceStyleSettings.CreateNaturalCharmingPreset();

        [SerializeField, TextArea(8, 20)]
        private string currentPrompt = string.Empty;

        public VoiceStyleSettings Settings => settings;
        public string CurrentPrompt => currentPrompt;

        public event Action<string> PromptChanged;

        private void Awake()
        {
            EnsureSettings();
            RebuildPrompt();
        }

        private void OnValidate()
        {
            EnsureSettings();
            RebuildPrompt();
        }

        public void RebuildPrompt()
        {
            EnsureSettings();
            settings.Clamp();
            currentPrompt = VoicePromptBuilder.Build(settings);
            PromptChanged?.Invoke(currentPrompt);
        }

        public void ApplyNaturalCharmingPreset()
        {
            settings = VoiceStyleSettings.CreateNaturalCharmingPreset();
            RebuildPrompt();
        }

        public void CopyPromptToClipboard()
        {
            GUIUtility.systemCopyBuffer = currentPrompt;
            Debug.Log("VoxCPM2 prompt copied to clipboard.");
        }

        private void EnsureSettings()
        {
            if (settings == null)
            {
                settings = VoiceStyleSettings.CreateNaturalCharmingPreset();
            }
        }
    }
}
