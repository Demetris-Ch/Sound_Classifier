package org.pytorch.demo.speechrecognition;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.gson.Gson;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;


public class MainActivity extends AppCompatActivity implements Runnable {
    private static final String TAG = MainActivity.class.getName();

    private Module mModuleEncoder;
    private Module mPreprocessEncoder;
    private TextView mTextView;
    private Button mButton;
    private Button sButton;
    private boolean forceStop = false;
    private final static int REQUEST_RECORD_AUDIO = 13;
    private final static int AUDIO_LEN_IN_SECOND = 2;
    private final static int SAMPLE_RATE = 44100;
    private final static int RECORDING_LENGTH = SAMPLE_RATE * AUDIO_LEN_IN_SECOND;

    private final static String LOG_TAG = MainActivity.class.getSimpleName();

    private int mStart = 1;
    private HandlerThread mTimerThread;
    private Handler mTimerHandler;
    private Runnable mRunnable = new Runnable() {
        @Override
        public void run() {
            mTimerHandler.postDelayed(mRunnable, 1000);

            MainActivity.this.runOnUiThread(
                    () -> {
                        mStart += 1;
                    });
        }
    };

    @Override
    protected void onDestroy() {
        stopTimerThread();
        super.onDestroy();
    }

    protected void stopTimerThread() {
        mTimerThread.quitSafely();
        try {
            mTimerThread.join();
            mTimerThread = null;
            mTimerHandler = null;
            mStart = 1;
        } catch (InterruptedException e) {
            Log.e(TAG, "Error on stopping background thread", e);
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButton = findViewById(R.id.btnRecognize);
        sButton = findViewById(R.id.stop_btn);
        mTextView = findViewById(R.id.tvResult);
        mButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {


                Thread thread = new Thread(MainActivity.this);
                thread.start();
    forceStop = false;
                mTimerThread = new HandlerThread("Timer");
                mTimerThread.start();
                mTimerHandler = new Handler(mTimerThread.getLooper());
                mTimerHandler.postDelayed(mRunnable, 1000);
                mButton.setEnabled(false);
                sButton.setEnabled(true);
                mTextView.setText("");
            }
        });

        sButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              forceStop = true;
                sButton.setEnabled(false);
                mButton.setEnabled(true);
            }
        });
        requestMicrophonePermission();
    }

    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }

    private void showTranslationResult(String result) {
        mTextView.setText(result);
    }

    public void run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();

        long shortsRead = 0;
        int recordingOffset = 0;
        short[] audioBuffer = new short[bufferSize / 2];
        short[] recordingBuffer = new short[RECORDING_LENGTH];

        while (!forceStop|| (shortsRead < 20)) {
            int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
            shortsRead += numberOfShort;
            System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberOfShort);
//            recordingOffset += numberOfShort;
        }

        record.stop();
        record.release();
        stopTimerThread();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mButton.setText("Recognizing...");
            }
        });

        float[] floatInputBuffer = new float[RECORDING_LENGTH];

        // feed in float values between -1.0f and 1.0f by dividing the signed 16-bit inputs.
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            floatInputBuffer[i] = recordingBuffer[i] / (float)Short.MAX_VALUE;
        }

        final String result = recognize(floatInputBuffer);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                showTranslationResult(result);
                mButton.setEnabled(true);
                mButton.setText("Start");
                sButton.setEnabled(false);
            }
        });
    }

    private String recognize(float[] floatInputBuffer) {

        double wav2vecinput[] = new double[RECORDING_LENGTH];
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            wav2vecinput[i] = floatInputBuffer[i];
        }
        Gson gson = new Gson();
        ApiRequest req = new ApiRequest();
        req.setValue(wav2vecinput);
        String json = gson.toJson(req);

        String result;

        try {
            String serverResponse = HttpController.post("https://audioclassifierflask.azurewebsites.net/api/audioclassifier", json);
            String results[] = serverResponse.split(":");
            result = results[0];



          //  double percentage = Double.valueOf(results[1]);
        } catch (IOException e) {
            result = "Server Error";
            e.printStackTrace();
        }
        return result;
    }
}