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
        CSVRecordReader rr = new CSVRecordReader(1, ';', '"');
        rr.initialize(new FileSplit(whiteWineFile));
        final List<List<Writable>> whiteWine = readAll(rr);

        // load the red wine data with a *new* CSV Record reader
        // You have to re-create the reader, as otherwise the header is not skipped, re-initializing is not enough
        rr = new CSVRecordReader(1, ';', '"');
        rr.initialize(new FileSplit(redWineFile));
        final List<List<Writable>> redWine = readAll(rr);

        // build two transform processes to add a constant integer column name "wine type",
        // with value 0 for white wine and value 1 for red
        final TransformProcess tpWhite = new TransformProcess.Builder(inputSchema)
                .addConstantIntegerColumn("wine type", 0)
                .build();
        final TransformProcess tpRed = new TransformProcess.Builder(inputSchema)
                .addConstantIntegerColumn("wine type", 1)
                .build();

        //process the red and white wine csvs with their respective transform process, then add the results
        // to one list
        final List<List<Writable>> allWineWithType = new ArrayList<>();
        allWineWithType.addAll(LocalTransformExecutor.execute(whiteWine, tpWhite));
        allWineWithType.addAll(LocalTransformExecutor.execute(redWine,   tpRed));

        //the red & white examples are now consecutive (first all white, then all red examples)
        //shuffle the list so they are in random order
        Collections.shuffle(allWineWithType);

        // split the list into testing, validation & training data, ratio 10:10:80
        final int splitpoint1 = allWineWithType.size() / 10;
        final int splitpoint2 = splitpoint1 * 2;
        List<List<Writable>> testing    = allWineWithType.subList(0, splitpoint1);
        List<List<Writable>> validation = allWineWithType.subList(splitpoint1, splitpoint2);
        List<List<Writable>> training   = allWineWithType.subList(splitpoint2, allWineWithType.size());

        //Use the *training* data to analyze the statistical properties of the columns using the AnalyzeLocal class.
        //You can use the CollectionRecordReader to wrap a list of Writables into a RecordReader
        final DataAnalysis dataAnalysisRaw = AnalyzeLocal.analyze(tpRed.getFinalSchema(), new CollectionRecordReader(training));
        System.out.println(dataAnalysisRaw);

        // build a new transform process for applying normalization to all columns except wine type & quality,
        // move wine type column to front, so quality is last again
        final TransformProcess tpNormalize = new TransformProcess.Builder(tpRed.getFinalSchema())
                .reorderColumns("wine type")
                .normalize("fixed acidity", Normalize.Standardize, dataAnalysisRaw)
                .normalize("volatile acidity", Normalize.Standardize, dataAnalysisRaw)
                .normalize("citric acid", Normalize.Standardize, dataAnalysisRaw)
                .normalize("residual sugar", Normalize.Standardize, dataAnalysisRaw)
                .normalize("chlorides", Normalize.Standardize, dataAnalysisRaw)
                .normalize("free sulfur dioxide", Normalize.Standardize, dataAnalysisRaw)
                .normalize("total sulfur dioxide", Normalize.Standardize, dataAnalysisRaw)
                .normalize("density", Normalize.Standardize, dataAnalysisRaw)
                .normalize("pH", Normalize.Standardize, dataAnalysisRaw)
                .normalize("sulphates", Normalize.Standardize, dataAnalysisRaw)
                .normalize("alcohol", Normalize.Standardize, dataAnalysisRaw)
                .build();

        // apply normalization transformation to each of the three datasets by using the LocalTransformExecutor again
        testing    = LocalTransformExecutor.execute(testing, tpNormalize);
        validation = LocalTransformExecutor.execute(validation, tpNormalize);
        training   = LocalTransformExecutor.execute(training, tpNormalize);

        // create an analysis of the final training data
        final DataAnalysis dataAnalysisProcessed = AnalyzeLocal.analyze(tpRed.getFinalSchema(), new CollectionRecordReader(training));
        System.out.println(dataAnalysisProcessed);

        // write each dataset to a new CSV File using the CSVRecordWriter (note that the output files have to exist first for writing)
        RecordWriter rw = new CSVRecordWriter();
        testingFile.createNewFile();
        rw.initialize(new FileSplit(testingFile), new NumberOfRecordsPartitioner());
        rw.writeBatch(testing);

        rw = new CSVRecordWriter();
        validationFile.createNewFile();
        rw.initialize(new FileSplit(validationFile), new NumberOfRecordsPartitioner());
        rw.writeBatch(validation);

        rw = new CSVRecordWriter();
        trainingFile.createNewFile();
        rw.initialize(new FileSplit(trainingFile), new NumberOfRecordsPartitioner());
        rw.writeBatch(training);

        // done!
        System.out.println("Preprocessing done!");
    }

}
