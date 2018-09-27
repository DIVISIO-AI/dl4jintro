package divisio.dl4jintro;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.AnalyzeLocal;
import org.datavec.local.transforms.LocalTransformExecutor;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Preprocessing {

    public static final Schema inputSchema = new Schema.Builder()
            // Input variables (based on physicochemical tests):
            // 1 - fixed acidity
            .addColumnDouble("fixed acidity")
            // 2 - volatile acidity
            .addColumnDouble("volatile acidity")
            // 3 - citric acid
            .addColumnDouble("citric acid")
            // 4 - residual sugar
            .addColumnDouble("residual sugar")
            // 5 - chlorides
            .addColumnDouble("chlorides")
            // 6 - free sulfur dioxide
            .addColumnDouble("free sulfur dioxide")
            // 7 - total sulfur dioxide
            .addColumnDouble("total sulfur dioxide")
            // 8 - density
            .addColumnDouble("density")
            // 9 - pH
            .addColumnDouble("pH")
            // 10 - sulphates
            .addColumnDouble("sulphates")
            // 11 - alcohol
            .addColumnDouble("alcohol")
            // Output variable (based on sensory data):
            // 12 - quality (score between 0 and 10)
            .addColumnInteger("quality")
            .build();
    //NOTE: do *not* use addColumnFloat - the column type cannot be analyzed and will cause an "Unknown column type: Float"
    // error

    /**
     * Utility method to read all data from a record reader into a list
     */
    private static List<List<Writable>> readAll(final RecordReader rr) {
        final ArrayList<List<Writable>> result = new ArrayList<>();
        while (rr.hasNext()) {
            result.add(rr.next());
        }
        return result;
    }

    /**
     * Merge the two wine files
     * Analyze the data
     * Normalize the data
     * Write resulting CSV Files for training, validation & testing
     */
    public static void main(final String[] args) throws Exception {
        //see https://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/basic/BasicDataVecExampleLocal.java
        //for examples of the API
        //see https://deeplearning4j.org/api/latest/
        //for the JavaDoc of the API

        //input files
        final File rawDataFolder = new File("raw_data");
        final File whiteWineFile = new File(rawDataFolder, "winequality-red.csv");
        final File redWineFile   = new File(rawDataFolder, "winequality-white.csv");

        //output files
        final File preprocessingFolder = new File("preprocessed_data");
        preprocessingFolder.mkdirs();
        final File trainingFile   = new File(preprocessingFolder, "training.csv");
        final File validationFile = new File(preprocessingFolder, "validation.csv");
        final File testingFile    = new File(preprocessingFolder, "testing.csv");

        //load white wine data with a CSV Record reader
        //CSVRecordReader rr = ...

        //we need a List<List<Writable>> to work without spar, use the readAll method to get such a list
        //final List<List<Writable>> whiteWine = ...

        // load the red wine data with a *new* CSV Record reader
        // You have to re-create the reader, as otherwise the header is not skipped, re-initializing is not enough
        //final List<List<Writable>> redWine = ...

        // build two transform processes to add a constant integer column name "wine type",
        // with value 0 for white wine and value 1 for red
        //final TransformProcess tpWhite = ...
        //final TransformProcess tpRed = ...

        //process the red and white wine csvs with their respective transform process and a LocalTransformExecutor,
        //then add the results to one list
        final List<List<Writable>> allWineWithType = new ArrayList<>();



        //the red & white examples are now consecutive (first all white, then all red examples)
        //shuffle the list so they are in random order


        // split the list into testing, validation & training data, ratio 10:10:80
        //List<List<Writable>> testing    = ...
        //List<List<Writable>> validation = ...
        //List<List<Writable>> training   = ...


        //Use the *training* data to analyze the statistical properties of the columns using the AnalyzeLocal class.
        //You can use the CollectionRecordReader to wrap a list of Writables into a RecordReader
        //final DataAnalysis dataAnalysisRaw = ...
        //System.out.println(dataAnalysisRaw);


        // build a new transform process for applying normalization to all columns except wine type & quality,
        // move wine type column to front, so quality is last again
        //final TransformProcess tpNormalize = new TransformProcess.Builder(tpRed.getFinalSchema())...


        // apply normalization transformation to each of the three datasets by using the LocalTransformExecutor again
        //testing    = ...
        //validation = ...
        //training   = ...


        // create an analysis of the final training data
        //final DataAnalysis dataAnalysisProcessed = ...
        //System.out.println(dataAnalysisProcessed);


        // write each dataset to a new CSV File using the CSVRecordWriter (note that the output files have to exist first for writing)
        // testingFile.createNewFile();
        // ...

        // done!
        //System.out.println("Preprocessing done!");
    }

}
