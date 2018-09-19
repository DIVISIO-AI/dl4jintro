package divisio.dl4jintro;

import java.io.File;

/**
 * Interface to hide different models / frameworks behind.
 * Very coarse, mainly meant to trigger certain actions and to make it easiert to write a general command line
 * application.
 */
public interface Trainer {

    /**
     * Let the training look for its last save state.
     * @param workingFolder the folder to save results in
     * @return if found, a file or subfolder with the last save state.
     */
    File findLastSaveState(final File workingFolder);

    /**
     * Initializes a brand new trainer.
     */
    void init();

    /**
     * Reinitializes the trainer from the given state.
     * @param saveState not null
     */
    void load(final File saveState);

    /**
     * Tell the trainer to save its state to the given folder.
     * @param workingFolder the folder to save results in
     * @return the file or folder created with the new save state.
     */
    File save(final File workingFolder);

    /**
     * Starts a new epoch, performs no training yet.
     * @return the number of the new epoch
     */
    int startEpoch();

    /**
     * Performs one iteration of training. How much that is depends on the trainer, it can be a quick minibatch or
     * a complete fit of the model.
     * @return true: training can continue with this epoch, false: training for this epoch is done
     */
    boolean train();

    /**
     * Trigger validation of the model.
     */
    void validate();
}
