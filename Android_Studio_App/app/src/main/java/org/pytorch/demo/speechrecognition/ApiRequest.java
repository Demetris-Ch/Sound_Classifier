package org.pytorch.demo.speechrecognition;

import java.io.Serializable;

public class ApiRequest {

    private double[] value;

    public double[] getValue() {
        return value;
    }

    public void setValue(double[] value) {
        this.value = value;
    }
}
