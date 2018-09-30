package divisio.dl4jintro;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A DL4J Multilayer trainer that trains a net to AND bits of two inputs together.
 */
public class BinaryOrTrainer extends AbstractDL4JMultilayerTrainer {

    private static final Logger log = LoggerFactory.getLogger(BinaryOrTrainer.class);

    /**
     * number of bits we want to OR
     */
    private final int bitCount = 24;

    /**
     * number of instances per mini-batch
     */
    private final int batchSize = 5;

    /**
     * number of instances we train on
     */
    private final int trainingSetSize = 2000;

    /**
     * number of instances we validate
     */
    private final int validationSetSize = 100;

    /**
     * Our data for training, will be wrapped in a DataSetIterator
     */
    private List<Pair<INDArray, INDArray>> trainingData = null;

    /**
     * Data for validating, separately created from training data, for large bitCounts there should be little to
     * no overlap with the training data.
     */
    private List<Pair<INDArray, INDArray>> validationData = null;

    public BinaryOrTrainer() {
        super(10);
    }

    @Override
    protected MultiLayerNetwork buildNetwork() {
        final MultiLayerConfiguration nnConf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Adam.builder().learningRate(0.01).build())
            .activation(Activation.RELU)
            .list(
                new OutputLayer.Builder(LossFunction.L2).nIn(bitCount * 2).nOut(bitCount)
                        .activation(Activation.SIGMOID).build()
            )
            .build();
        return new MultiLayerNetwork(nnConf);
    }

    /**
     * Builds an AND test instance with the currently configured number of bits
     * @param rnd the random generator to use, not null
     * @return a test instance with two sets of bits as input and two sets of bits as output
     */
    public Pair<INDArray, INDArray> buildInstance(final Random rnd) {
        final double[] input = new double[bitCount * 2];
        final double[] labels = new double[bitCount];
        for (int idx = 0; idx < bitCount; ++idx) {
            boolean bitA = rnd.nextBoolean();
            boolean bitB = rnd.nextBoolean();
            boolean result = bitA || bitB;
            input[idx]            = bitA   ? 1.0 : 0.0;
            input[idx + bitCount] = bitB   ? 1.0 : 0.0;
            labels[idx]           = result ? 1.0 : 0.0;
        }
        return Pair.create(
            Nd4j.create(input),
            Nd4j.create(labels)
        );
    }

    /**
     * Builds the given number of training instances
     * @param numberOfInstances
     * @return a list with instances, never null
     */
    public List<Pair<INDArray, INDArray>> buildData(final int numberOfInstances) {
        final Random rnd = new Random();
        final ArrayList<Pair<INDArray, INDArray>> result = new ArrayList<>(numberOfInstances);
        for (int i = 0; i < numberOfInstances; ++i) {
            result.add(buildInstance(rnd));
        }
        return result;
    }

    @Override
    protected DataSetIterator buildIterator() {
        if (trainingData == null) {
            trainingData = buildData(trainingSetSize);
        }
        return new INDArrayDataSetIterator(trainingData, batchSize);
    }

    @Override
    public void validate() {
        if (validationData == null) {
            validationData = buildData(validationSetSize);
        }

        final ROCMultiClass evaluationResult =
                nn.evaluateROCMultiClass(new INDArrayDataSetIterator(validationData, 1));

        log.info("\n" + evaluationResult.stats());

        //manually print a couple of examples
        final int maxExamples = 5;
        int counter = 0;
        log.info("\n Raw outputs: ");
        for (final Pair<INDArray, INDArray> inputOutput : validationData) {
            final INDArray input  = inputOutput.getFirst();
            final INDArray output = inputOutput.getSecond();
            final INDArray prediction = nn.output(input);
            log.info("\n" + input + "\n" + prediction + "\n" + output);

            ++counter;
            if (counter >= maxExamples) {
                break;
            }
        }
    }
}
