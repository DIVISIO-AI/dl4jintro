package divisio.dl4jintro;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Abstract base class that provides common code for training a DL4J multilayer network.
 */
public abstract class AbstractDL4JMultilayerTrainer implements Trainer {

    //constants for the save file name
    private static final String SAVE_FILE_PREFIX = "multilayer";
    private static final String SAVE_FILE_SUFFIX = ".zip";
    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");

    /** the number of iterations between printing the score */
    protected final int printIterations;
    /** the network we are training */
    protected MultiLayerNetwork nn;
    /** the iterator providing the training data */
    protected DataSetIterator trainingIterator;

    public AbstractDL4JMultilayerTrainer(final int printIterations) {
        this.printIterations = printIterations;
    }

    /**
     * Implemented by subclasses, defines which network to train.
     */
    protected abstract MultiLayerNetwork buildNetwork();

    /**
     * Implemented by subclasses, defines how to load data.
     */
    protected abstract DataSetIterator buildIterator();

    /**
     * @return a filename for a savestate, contains data, epoch & iteration
     */
    protected String buildSaveFilename() {
        final String now = LocalDateTime.now().format(DATE_TIME_FORMATTER);
        final String epoch = Integer.toString(nn.getEpochCount());
        final String batchCount = Integer.toString(nn.getIterationCount());
        return SAVE_FILE_PREFIX + "_" + now + "_" + epoch + "_" + batchCount + SAVE_FILE_SUFFIX;
    }

    /**
     * attaches listeners to the network that monitor training progress
     */
    protected void attachListeners() {
        nn.setListeners(new ScoreIterationListener(printIterations));//logs scores during training
    }

    @Override
    public File findLastSaveState(final File workingFolder) {
        final File[] children = workingFolder.listFiles();
        if (children == null) { return null; }
        final SortedSet<File> saveStates = new TreeSet<>();
        for (final File child : children) {
            if (child.isFile() &&
                child.canRead() &&
                child.getName().startsWith(SAVE_FILE_PREFIX) &&
                child.getName().endsWith(SAVE_FILE_SUFFIX))
            {
                saveStates.add(child);

            }
        }
        if (saveStates.isEmpty()) { return null; }
        return saveStates.last();
    }

    @Override
    public void init() {
        nn = buildNetwork();
        attachListeners();
    }

    @Override
    public void load(final File saveState) {
        try {
            nn = MultiLayerNetwork.load(saveState, true);
            attachListeners();
        } catch (final IOException e) {
            throw new RuntimeException("Could not load MultiLayerNetwork save state from: " + saveState, e);
        }
    }

    @Override
    public File save(final File workingFolder) {
        final File saveFile = new File(workingFolder, buildSaveFilename());
        try {
            nn.save(saveFile, true);
        } catch (final IOException e) {
            throw new RuntimeException("Could not save MultiLayerNetwork save state to: " + saveFile, e);
        }
        return saveFile;
    }

    @Override
    public int startEpoch() {
        //if we do not have a data set iterator yet, or if we have one that cannot be reset, build a new one
        if (trainingIterator == null || !trainingIterator.resetSupported()) {
            trainingIterator = buildIterator();
        } else {
            //if we can just reset the iterator, do that, it is generally cheaper
            trainingIterator.reset();
        }
        //increase counter so we know what iteration we are in
        nn.incrementEpochCount();
        return nn.getEpochCount();
    }

    @Override
    public boolean train() {
        if (trainingIterator.hasNext()) {
            nn.fit(trainingIterator.next());
        }
        return trainingIterator.hasNext();
    }
}
