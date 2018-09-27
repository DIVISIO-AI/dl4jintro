package divisio.dl4jintro;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A DL4J Multilayer trainer that trains a net to AND bits of two inputs together.
 */
public class WineTrainer extends AbstractDL4JMultilayerTrainer {

    private static final Logger log = LoggerFactory.getLogger(WineTrainer.class);

    /**
     * number of input columns
     */
    private final int nInputFeatures = 12;

    /**
     * the output feature is in the last column
     */
    private final int idxOutputFeature = nInputFeatures;

    /**
     * number of output classes (scores are 1-10)
     */
    private final int nOutputFeatures = 10;

    /**
     * number of instances per mini-batch
     */
    private final int batchSize = 50;


    public WineTrainer() {
        super(500);
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
                new DenseLayer.Builder().nIn(nInputFeatures).nOut(nInputFeatures).build(),
                new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nIn(nInputFeatures).nOut(nOutputFeatures)
                        .activation(Activation.SOFTMAX).build()
            )
            .build();
        return new MultiLayerNetwork(nnConf);
    }

    private DataSetIterator buildIterator(final File csvFile, final int batchSize) {
        try {
            final RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(csvFile));
            return new RecordReaderDataSetIterator(
                    rr, null, batchSize,
                    idxOutputFeature, idxOutputFeature, nOutputFeatures, -1,
                    false);
        } catch (final Exception ioe) {
            throw new RuntimeException("Could not build RecordReader for: " + csvFile, ioe);
        }
    }

    @Override
    protected DataSetIterator buildIterator() {
        return buildIterator(new File("preprocessed_data/training.csv"), batchSize);
    }

    @Override
    public void validate() {
        final DataSetIterator validationDataIterator = buildIterator(new File("preprocessed_data/validation.csv"), 1);

        final Evaluation evaluationResult = nn.evaluate(validationDataIterator);

        log.info("\n" + evaluationResult.stats());
    }
}
