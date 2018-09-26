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
public class BinaryAndTrainer extends AbstractDL4JMultilayerTrainer {

    private static final Logger log = LoggerFactory.getLogger(BinaryAndTrainer.class);

    /**
     * number of bits we want to AND
     */
    private final int bitCount = 1;

    /**
     * number of instances per mini-batch
     */
    private final int batchSize = 5;

    /**
     * number of instances we train on
     */
    private final int trainingSetSize = 20;

    /**
     * number of instances we validate
     */
    private final int validationSetSize = 10;

    /**
     * Our data for training, will be wrapped in a DataSetIterator
     */
    private List<Pair<INDArray, INDArray>> trainingData = null;

    /**
     * Data for validating, separately created from training data, for large bitCounts there should be little to
     * no overlap with the training data.
     */
    private List<Pair<INDArray, INDArray>> validationData = null;

    public BinaryAndTrainer() {
        super(10);
    }

    @Override
    protected MultiLayerNetwork buildNetwork() {
        final MultiLayerConfiguration nnConf = new NeuralNetConfiguration.Builder()
            .seed(679876471)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Adam.builder().learningRate(0.01).build())
            .activation(Activation.RELU)
            .list(
                new DenseLayer.Builder().nIn(bitCount * 2).nOut(bitCount * 2).build(),
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
            boolean result = bitA && bitB;
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
    }
}
