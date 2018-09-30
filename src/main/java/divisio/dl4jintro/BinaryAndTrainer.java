package divisio.dl4jintro;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
import java.util.Collections;
import java.util.List;

/**
 * A DL4J Multilayer trainer that trains a net to AND bits of two inputs together.
 */
public class BinaryAndTrainer extends AbstractDL4JMultilayerTrainer {

    private static final Logger log = LoggerFactory.getLogger(BinaryAndTrainer.class);

    /**
     * number of instances per mini-batch
     */
    private final int batchSize = 1; //try 1 and 4

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
        super(1);
    }

    @Override
    protected MultiLayerNetwork buildNetwork() {
        final MultiLayerConfiguration nnConf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Adam.builder().learningRate(0.01).build())
            .activation(Activation.RELU)
            .list(
                new OutputLayer.Builder(LossFunction.L2).nIn(2).nOut(1)
                        .activation(Activation.SIGMOID).build()
            )
            .build();
        return new MultiLayerNetwork(nnConf);
    }

    /**
     * Builds an AND test instance with the given boolean values
     * @param bitA the first input
     * @param bitB the second input
     * @return a test instance with two sets of bits as input and two sets of bits as output
     */
    public Pair<INDArray, INDArray> buildInstance(final boolean bitA, final boolean bitB) {
        final double[] input = new double[2];
        final double[] labels = new double[1];
        for (int idx = 0; idx < labels.length; ++idx) {
            final boolean result = bitA && bitB;
            input[idx * 2]     = bitA   ? 1.0 : 0.0;
            input[idx * 2 + 1] = bitB   ? 1.0 : 0.0;
            labels[idx]        = result ? 1.0 : 0.0;
        }
        return Pair.create(
            Nd4j.create(input),
            Nd4j.create(labels)
        );
    }

    /**
     * Builds all combinations of instances*
     * @return a list with instances, never null
     */
    public List<Pair<INDArray, INDArray>> buildData() {
        final ArrayList<Pair<INDArray, INDArray>> result = new ArrayList<>(4);

        for (final boolean bitA : new boolean[]{true, false}) {
            for (final boolean bitB : new boolean[]{true, false}) {
                result.add(buildInstance(bitA, bitB));
            }
        }

        return result;
    }

    @Override
    protected DataSetIterator buildIterator() {
        if (trainingData == null) {
            trainingData = buildData();
            //shuffle the data, so we get a different order between resumed trainings - helps a bit escaping
            //when the network is "stuck"
            Collections.shuffle(trainingData);
        }
        return new INDArrayDataSetIterator(trainingData, batchSize);
    }

    @Override
    public void validate() {
        if (validationData == null) {
            validationData = buildData();
        }

        final ROCMultiClass evaluationResult =
                nn.evaluateROCMultiClass(new INDArrayDataSetIterator(validationData, 1));

        log.info("\n" + evaluationResult.stats());

        //manually print a couple of examples
        final int maxExamples = 4;
        int counter = 0;
        log.info("\n Raw outputs: ");
        for (final Pair<INDArray, INDArray> inputOutput : validationData) {
            final INDArray input  = inputOutput.getFirst();
            final INDArray output = inputOutput.getSecond();
            final INDArray prediction = nn.output(input);
            log.info(input + " -> " + prediction + ", training data: " + output);

            ++counter;
            if (counter >= maxExamples) {
                break;
            }
        }
    }
}
