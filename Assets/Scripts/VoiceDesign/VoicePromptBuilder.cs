using System;
using System.Collections.Generic;

namespace VCBM.VoiceDesign
{
    public static class VoicePromptBuilder
    {
        public const string NaturalCharmingReference =
            "A Japanese young adult woman speaking in a relaxed everyday conversation. " +
            "Her voice is light, clear, warm and naturally charming, " +
            "with a gentle smile and soft vocal resonance. " +
            "She speaks at a natural pace with subtle, varied intonation. " +
            "Realistic and human, without exaggerated acting or an excessively high pitch.";

        public static string Build(VoiceStyleSettings source)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }

            VoiceStyleSettings s = source.Copy();
            s.Clamp();

            var parts = new List<string>
            {
                BuildSpeaker(s.youthful),
                BuildPitch(s.pitch),
                BuildTone(s.brightness, s.softness),
                BuildBreathiness(s.breathiness),
                BuildCharacter(s.cuteness, s.energy),
                BuildIntonation(s.expressiveness),
                BuildSpeed(s.speed),
                BuildNaturalness(s.naturalness)
            };

            string custom = NormalizeSentence(s.customInstruction);
            if (!string.IsNullOrEmpty(custom))
            {
                parts.Add(custom);
            }

            return string.Join(" ", parts);
        }

        private static string BuildSpeaker(float value)
        {
            if (value >= 0.80f)
            {
                return "A youthful Japanese young woman speaking casually in a relaxed everyday conversation.";
            }

            if (value >= 0.40f)
            {
                return "A Japanese young adult woman speaking in a relaxed everyday conversation.";
            }

            return "A mature Japanese woman speaking calmly in a relaxed everyday conversation.";
        }

        private static string BuildPitch(float value)
        {
            if (value >= 0.82f)
            {
                return "Her voice has a distinctly high, light and youthful register that remains comfortable and effortless.";
            }

            if (value >= 0.64f)
            {
                return "Her voice has a naturally elevated, light register without sounding strained.";
            }

            if (value <= 0.28f)
            {
                return "Her voice has a calm, slightly low and stable register.";
            }

            return "Her voice stays in a comfortable, naturally light register.";
        }

        private static string BuildTone(float brightness, float softness)
        {
            string tone;

            if (brightness >= 0.68f)
            {
                tone = "Her vocal tone is light, clear, fresh and bright";
            }
            else if (brightness <= 0.32f)
            {
                tone = "Her vocal tone is clear, warm, calm and mellow";
            }
            else
            {
                tone = "Her vocal tone is light, clear and warm";
            }

            if (softness >= 0.68f)
            {
                return tone + ", with a gentle smile and soft, rounded vocal resonance.";
            }

            if (softness <= 0.32f)
            {
                return tone + ", with focused resonance and crisp articulation.";
            }

            return tone + ", with a gentle smile and balanced vocal resonance.";
        }

        private static string BuildBreathiness(float value)
        {
            if (value >= 0.72f)
            {
                return "A slight airy quality adds softness while the speech remains clear and intelligible.";
            }

            if (value <= 0.30f)
            {
                return "The voice remains clean and focused without excessive breathiness.";
            }

            return "A very subtle airy texture keeps the voice soft without reducing clarity.";
        }

        private static string BuildCharacter(float cuteness, float energy)
        {
            string character;

            if (cuteness >= 0.78f)
            {
                character = "She sounds naturally charming, sweet and slightly playful";
            }
            else if (cuteness >= 0.45f)
            {
                character = "She sounds naturally charming and friendly";
            }
            else
            {
                character = "She sounds composed and understated";
            }

            if (energy >= 0.72f)
            {
                return character + ", with cheerful and pleasantly lively energy.";
            }

            if (energy <= 0.30f)
            {
                return character + ", with a calm, gentle and relaxed mood.";
            }

            return character + ", with relaxed but attentive energy.";
        }

        private static string BuildIntonation(float value)
        {
            if (value >= 0.72f)
            {
                return "She uses lively but natural variations in rhythm and intonation.";
            }

            if (value <= 0.30f)
            {
                return "She uses restrained, calm and stable intonation.";
            }

            return "She uses subtle, varied intonation and small spontaneous changes in rhythm.";
        }

        private static string BuildSpeed(float value)
        {
            if (value >= 0.72f)
            {
                return "She speaks at a slightly quick but comfortable conversational pace.";
            }

            if (value <= 0.30f)
            {
                return "She speaks slowly and gently with natural pauses.";
            }

            return "She speaks at a natural conversational pace.";
        }

        private static string BuildNaturalness(float value)
        {
            if (value >= 0.82f)
            {
                return "The performance is realistic, human and conversational, without forced pitch, exaggerated acting, squeaking or cartoon-like delivery.";
            }

            if (value >= 0.52f)
            {
                return "The performance is natural and conversational without excessive acting.";
            }

            return "The performance may be slightly stylized while remaining clear and coherent.";
        }

        private static string NormalizeSentence(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return string.Empty;
            }

            string trimmed = text.Trim();
            char last = trimmed[trimmed.Length - 1];

            if (last == '.' || last == '!' || last == '?' ||
                last == 'ÅB' || last == 'ÅI' || last == 'ÅH')
            {
                return trimmed;
            }

            return trimmed + ".";
        }
    }
}
