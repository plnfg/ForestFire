
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Pelin
 */
public final class ANN {

    final float lr = 0.4f;
    final float mom = 0.3f;
    final int HIDDEN_SIZE = 3;
    final int OUTP_SIZE = 1;

    float[][] inputW;
    float[][] hiddenW;
    float[] actual;
    Node[][] inputNode;
    Node[][] hiddenNode;
    Node[] outpNode;
    float RMSE;
    float min;
    float[][] best_iW;
    float[][] best_hW;
    float sumRMSE;
    Random r = new Random(99);

    public static void main(String[] args) {
        new ANN().Train_Test_Data();
    }

    public void Train_Test_Data() {
        ReadArea();
        ReadDataSet();
        //
        init();
        startEpochs(88, 440);
        testData(0, 88);

        init();
        startEpochs(0, 88);
        startEpochs(176, 440);
        testData(88, 176);
        //
        init();
        startEpochs(0, 176);
        startEpochs(264, 440);
        testData(176, 264);
        //
        init();
        startEpochs(0, 264);
        startEpochs(352, 440);
        testData(264, 352);
        //
        init();
        startEpochs(0, 352);
        testData(352, 440);

        System.out.println("Avg.RMSE: "+String.valueOf(sumRMSE / 5));

    }

    public void startEpochs(int init, int end) {
        int i, k, l;

        for (i = 0; i < 500; i++) {
            FeedForward(init, end);
            BackPropagation(init, end);
            if (RMSE < min) {
                min = RMSE;
                for (k = 0; k < HIDDEN_SIZE; k++) {
                    for (l = 0; l < 15; l++) {
                        best_iW[l][k] = inputW[l][k];
                    }
                    best_hW[k][0] = hiddenW[k][0];
                }
            }
            RMSE = 0;
        }

    }

    public void testData(int init, int end) {
        int i, j;
        for (i = 0; i < HIDDEN_SIZE; i++) {

            hiddenW[i][0] = best_hW[i][0];
            for (j = 0; j < 15; j++) {
                inputW[j][i] = best_iW[j][i];
            }
        }

        FeedForward(init, end);
        sumRMSE += RMSE;
//        System.out.println(String.valueOf(min));
//        System.out.println(String.valueOf(RMSE));
//        System.out.println("--------------- ");
    }

    public void FeedForward(int init, int end) {
        int i, j, k;
        float input_weight_sum = 0;
        float hidden_weight_sum = 0;
        float net, output;
        float SSE = 0;

        for (k = init; k < end; k++) {
            for (j = 0; j < HIDDEN_SIZE; j++) {
                for (i = 0; i < 15; i++) {
                    input_weight_sum += (inputW[i][j] * inputNode[k][i].getInput());
                }
                net = input_weight_sum + hiddenNode[k][j].getBias();
                hiddenNode[k][j].setInput(net);
                output = activation(net);
                hiddenNode[k][j].setOutput(output);
                hidden_weight_sum += hiddenNode[k][j].getOutput() * hiddenW[j][0];
                input_weight_sum = 0;

            }
            net = hidden_weight_sum + outpNode[k].getBias();
            outpNode[k].setInput(net);
            output = activation(net);
            outpNode[k].setOutput(output);
            
            SSE += pow((denormalize(actual[k]) - denormalize(output)), 2);

            hidden_weight_sum = 0;
        }

        RMSE = (float) Math.sqrt(SSE / 440);

    }

    public void BackPropagation(int init, int end) {

        int i = r.nextInt(end - init) + init;
        float delta;
        float[][] new_hW = new float[HIDDEN_SIZE][1];
        float[][] new_iW = new float[15][HIDDEN_SIZE];
        float hid_node_err;
        float output = outpNode[i].getOutput();
        // System.out.println(String.valueOf(i));
        //System.out.println(String.valueOf(output));
        //System.out.println(String.valueOf(actual[i]));
        float outp_node_err = output * (1 - output) * (actual[i] - output);

        int k, l;
        float bias = outpNode[i].getBias();
        float new_bias = (lr * outp_node_err * 1 * mom) + bias;
        
        for (k = 0; k < HIDDEN_SIZE; k++) {
            delta = lr * mom * hiddenNode[i][k].getOutput() * outp_node_err;
            new_hW[k][0] = hiddenW[k][0] + delta;
        }

        for (k = 0; k < HIDDEN_SIZE; k++) {
            output = hiddenNode[i][k].getOutput();
            hid_node_err = output * (1 - output) * (hiddenW[k][0] * outp_node_err);
            for (l = 0; l < 15; l++) {
                delta = (hid_node_err) * lr * mom * inputNode[i][l].getInput();
                new_iW[l][k] = inputW[l][k] + delta;
                inputW[l][k] = new_iW[l][k];
            }
            new_bias = (lr * mom * hid_node_err) + hiddenNode[i][k].getBias();
            hiddenNode[i][k].setBias(new_bias);
            hiddenW[k][0] = new_hW[k][0];
        }

        for (k = init; k < end; k++) {
            outpNode[k].setBias(new_bias);
            for (l = 0; l < HIDDEN_SIZE; l++) {
                hiddenNode[k][l].setBias(hiddenNode[i][l].getBias());
            }
        }

    }

    public void ReadArea() {
        try {
            actual = new float[440];
            int i = 0;
            Path filePath1 = Paths.get("area.txt");
            Scanner scanner = new Scanner(filePath1);
            while (scanner.hasNext()) {
                if (scanner.hasNextLine()) {
                    actual[i] = (scanner.nextFloat());
                } else {
                    scanner.next();
                }
                i++;
            }

        } catch (IOException ex) {
            Logger.getLogger(ANN.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    public void ReadDataSet() {
        input_node_init();
        try {
            String line;
            int i = 0, j ;
            String FOREST = "dataset1.txt";
            File file_forest = new File(FOREST);

            try (BufferedReader reader = new BufferedReader(new FileReader(file_forest))) {
                while ((line = reader.readLine()) != null) {
                    StringTokenizer st = new StringTokenizer(line, ";");

                    for (j = 0; j < 15; j++) {
                        inputNode[i][j].setInput(Float.valueOf(st.nextToken()));
                    }
                    i++;
                }

            }

        } catch (IOException | NumberFormatException e) {
            System.out.println(e);

        }

    }

    float activation(float x) {
        return (float) (1 / (1 + exp((-1) * x)));
    }

    public float denormalize(float x) {
        return (float) (x * (1090.84 - 0)) + 0;
    }

    public void input_node_init() {
        inputNode = new Node[440][15];
        int i , j ;
        for (i = 0; i < 440; i++) {
            for (j = 0; j < 15; j++) {
                inputNode[i][j] = new Node();
            }

        }
    }

    public void hidden_node_init() {
        hiddenNode = new Node[440][HIDDEN_SIZE];
        int i, j;
        float bias = (float) 0.3;
        for (i = 0; i < 440; i++) {
            for (j = 0; j < HIDDEN_SIZE; j++) {
                hiddenNode[i][j] = new Node();
                hiddenNode[i][j].setBias(bias);
            }

        }
    }

    public void outp_node_init() {
        outpNode = new Node[440];
        int i;
        float bias = (float) 0.2;
        for (i = 0; i < 440; i++) {
            outpNode[i] = new Node();
            outpNode[i].setBias(bias);
        }

    }

    public void weight_init() {
        inputW = new float[15][HIDDEN_SIZE];
        int i, j;
        for (i = 0; i < 15; i++) {
            for (j = 0; j < HIDDEN_SIZE; j++) {
                inputW[i][j] = r.nextFloat() - 0.5f;
            }
        }
        hiddenW = new float[HIDDEN_SIZE][OUTP_SIZE];
        for (i = 0; i < HIDDEN_SIZE; i++) {
            for (j = 0; j < OUTP_SIZE; j++) {
                hiddenW[i][j] = r.nextFloat() - 0.5f;
            }
        }

    }

    public void init() {
        weight_init();
        hidden_node_init();
        outp_node_init();
        best_iW = new float[15][HIDDEN_SIZE];
        best_hW = new float[HIDDEN_SIZE][1];
        RMSE = 0;
        min = 999999999;
    }
}
