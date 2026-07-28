using System;

namespace VCBM.VoxCPM
{
    [Serializable]
    public sealed class VoxCpmGenerateRequest
    {
        public string voice_prompt;
        public string speech_text;
        public float cfg_value;
        public int inference_timesteps;
        public int seed;
        public int candidate_count;
        public bool normalize;
    }

    [Serializable]
    public sealed class VoxCpmCandidate
    {
        public string candidate_id;
        public string audio_url;
        public string file_name;
        public int sample_rate;
        public int seed;
        public float duration_seconds;
    }

    [Serializable]
    public sealed class VoxCpmGenerateResponse
    {
        public bool success;
        public string request_id;
        public VoxCpmCandidate[] candidates;
        public string error;
    }

    [Serializable]
    public sealed class VoxCpmHealthResponse
    {
        public bool success;
        public string service;
        public string model_state;
        public string model_id;
        public string device;
        public string error;
    }
}
