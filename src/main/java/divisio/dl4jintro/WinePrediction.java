package divisio.dl4jintro;


import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.converters.FileConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * Command line application that manually runs each test instance and additionally runs a validation with the test data.
 */
public class WinePrediction {

    @Parameter(names = {"-h", "--help"},
            description = "Show usage info.",
            help = true)
    private boolean help;

    @Parameter(names = {"-wf", "--workingFolder"},
            description = "Folder for loading / saving models, log file, etc.",
            converter = FileConverter.class,
            required = true)
    private File workingFolder;

    @Parameter(names = {"-if", "--inputFile"},
            description = "CSV File with values to predict.",
            converter = FileConverter.class,
            required = true)
    private File inputFile;

    /**
     * Loads the network for predictions
     */
    private MultiLayerNetwork loadNetwork() throws IOException {
        final WineTrainer wineTrainer = new WineTrainer();
        final File lastSave = wineTrainer.findLastSaveState(workingFolder);
        if (lastSave == null) {
            throw new IOException("No save state found in folder " + workingFolder);
        }
        //note the only difference from the code in the Abstract Trainer base class: we do not need to load the updater
        //(the updater contains information about the current state of the learning process)
        return MultiLayerNetwork.load(lastSave, false);
    }

    /**
     * manually runs each test instance through the network, similar to what needs to be done in production
     */
    private void runTestManual() throws IOException, InterruptedException {
        //restore network
        final MultiLayerNetwork nn = loadNetwork();
        //build reader for CSV
        final RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(inputFile));
        //wrap reader into a data set iterator
        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(
                rr, null, 1,
                WineTrainer.idxOutputFeature, WineTrainer.idxOutputFeature, WineTrainer.nOutputFeatures, -1,
                false);
        //now we manually loop over the data set iterator and create predictions - in real live, you would just wrap
        //your input data manually into a DataSet or INDArray. But always take care that that data gets the EXACT SAME
        //PREPROCESSING AS YOUR TRAINING DATA. This is the most important thing about taking a model live and needs to
        //be addressed with great care and tested well. We preprocessed everything when we wrote the testing CSV, so
        //there is no need to do anything
        while (dataSetIterator.hasNext()) {
            final DataSet dataSet = dataSetIterator.next();
            //this time, we need only the features, as there is no learning - we just want the result
            //final INDArray features =

            //perform the prediction using the Neural Networks output function
            // - note that in production you need to make sure this is synchronized, as the network
            //is not thread-safe
            //final INDArray prediction =

            //get the output index with the highest value using the argMax function - that is our wine score
            //(the result of the network is an array of probabilities for each class)
            //TIPP: use the debugger to inspect the resulting array and determine the axis
            //final INDArray maxIdx =

            //get the int out of the INDArray with getInt
            //final int predictedClass =

            //get the expected class from the dataSet in the same way, using getLabels, argMax and getInt
            //final int expectedClass =

            //we simply print the result here - for real test scenarious, e.g. image classification you might also print
            //the original file name etc.
            //final boolean correct = predictedClass == expectedClass;
            //System.out.println((correct ? "Y " : "N ") + predictedClass + ", " + expectedClass);
        }
    }

    private void runTestStatistics() throws IOException {
        final WineTrainer wineTrainer = new WineTrainer();
        final File lastSave = wineTrainer.findLastSaveState(workingFolder);
        if (lastSave == null) {
            throw new IOException("No save state found in folder " + workingFolder);
        }
        wineTrainer.load(lastSave);
        wineTrainer.validate(inputFile);
    }

    public static void main(final String[] args) throws Exception {
        // create instance of our Console Application
        final WinePrediction app = new WinePrediction();

        // parse command line params
        final JCommander commander = JCommander.newBuilder().addObject(app).build();
        try {
            commander.parse(args);
        } catch (final ParameterException pe) {
            //thrown if the given arguments are invalid - print the error message, print usage instructions & exit.
            System.out.println(pe.getMessage());
            commander.usage();
            System.exit(-1);
            return;
        }
        if (app.help) {
            commander.usage();
            System.exit(0);
            return;
        }

        app.runTestManual();
        app.runTestStatistics();
    }
}
