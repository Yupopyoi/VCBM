using System;
using System.Diagnostics;
using System.Text;

#if UNITY_EDITOR
using UnityEditor;
#endif

public static class LocalWavFilePicker
{
    public static string Open()
    {
#if UNITY_EDITOR
        return EditorUtility.OpenFilePanel(
            "基にするWAV音声を選択",
            "",
            "wav"
        );
#elif UNITY_STANDALONE_WIN
        return OpenWindowsDialog();
#else
        return string.Empty;
#endif
    }

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
    private static string OpenWindowsDialog()
    {
        const string script =
            "Add-Type -AssemblyName System.Windows.Forms; " +
            "$dialog = New-Object System.Windows.Forms.OpenFileDialog; " +
            "$dialog.Title = '基にするWAV音声を選択'; " +
            "$dialog.Filter = 'WAV files (*.wav)|*.wav'; " +
            "$dialog.Multiselect = $false; " +
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) " +
            "{ [Console]::Out.Write($dialog.FileName) }";

        string encodedCommand = Convert.ToBase64String(
            Encoding.Unicode.GetBytes(script)
        );

        var startInfo = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = $"-NoProfile -STA -EncodedCommand {encodedCommand}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        using (Process process = Process.Start(startInfo))
        {
            if (process == null)
            {
                return string.Empty;
            }

            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            return process.ExitCode == 0
                ? output.Trim()
                : string.Empty;
        }
    }
#endif
}
