using System;
using UnityEngine;

namespace VCBM.VoiceDesign
{
    [Serializable]
    public sealed class VoiceStyleSettings
    {
        [Range(0f, 1f)] public float youthful = 0.55f;
        [Range(0f, 1f)] public float pitch = 0.55f;
        [Range(0f, 1f)] public float brightness = 0.65f;
        [Range(0f, 1f)] public float softness = 0.70f;
        [Range(0f, 1f)] public float breathiness = 0.25f;
        [Range(0f, 1f)] public float energy = 0.40f;
        [Range(0f, 1f)] public float cuteness = 0.60f;
        [Range(0f, 1f)] public float expressiveness = 0.55f;
        [Range(0f, 1f)] public float speed = 0.50f;
        [Range(0f, 1f)] public float naturalness = 0.90f;

        [TextArea(2, 5)]
        public string customInstruction = string.Empty;

        public static VoiceStyleSettings CreateNaturalCharmingPreset()
        {
            return new VoiceStyleSettings
            {
                youthful = 0.55f,
                pitch = 0.55f,
                brightness = 0.65f,
                softness = 0.70f,
                breathiness = 0.25f,
                energy = 0.40f,
                cuteness = 0.60f,
                expressiveness = 0.55f,
                speed = 0.50f,
                naturalness = 0.90f,
                customInstruction = string.Empty
            };
        }

        public VoiceStyleSettings Copy()
        {
            return new VoiceStyleSettings
            {
                youthful = youthful,
                pitch = pitch,
                brightness = brightness,
                softness = softness,
                breathiness = breathiness,
                energy = energy,
                cuteness = cuteness,
                expressiveness = expressiveness,
                speed = speed,
                naturalness = naturalness,
                customInstruction = customInstruction ?? string.Empty
            };
        }

        public void Clamp()
        {
            youthful = Mathf.Clamp01(youthful);
            pitch = Mathf.Clamp01(pitch);
            brightness = Mathf.Clamp01(brightness);
            softness = Mathf.Clamp01(softness);
            breathiness = Mathf.Clamp01(breathiness);
            energy = Mathf.Clamp01(energy);
            cuteness = Mathf.Clamp01(cuteness);
            expressiveness = Mathf.Clamp01(expressiveness);
            speed = Mathf.Clamp01(speed);
            naturalness = Mathf.Clamp01(naturalness);

            if (customInstruction == null)
            {
                customInstruction = string.Empty;
            }
        }
    }
}
