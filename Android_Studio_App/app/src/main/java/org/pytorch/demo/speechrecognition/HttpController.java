package org.pytorch.demo.speechrecognition;

import java.io.IOException;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class HttpController {

    public static final MediaType JSON
            = MediaType.parse("application/json; charset=utf-8");
   public static  String post(String url, String json) throws IOException {

       OkHttpClient client = new OkHttpClient();
        RequestBody body = RequestBody.create(JSON, json); // new
        Request request = new Request.Builder()
                .url(url)
                .post(body)
                .build();
        Response response = client.newCall(request).execute();
        return response.body().string();
    }
}