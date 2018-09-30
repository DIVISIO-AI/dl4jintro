# Example Project for the DIVISIO DL4J Intro Workshop

## Prerequisites

You will need the following software installed: 

 * git
 * JDK 8 or higher
 * Maven 3
 * a JAVA Editor or IDE of your choice, presentations will be done with IntelliJ

## Prepare for the workshop

Make sure the above software is installed.

To prepare for the workshop, please clone this repository: 

    git clone https://github.com/DIVSIO/dl4jintro.git
    
Build the project once to do download all dependencies, sources & javadoc. DL4J has a lot of dependencies, 
so it is better if you downloaded them before the workshop.

    mvn clean package -DdownloadSources=true -DdownloadJavadocs=true
    
If you want, you can run the application once like so: 

    mvn exec:java -Dexec.mainClass="divisio.dl4jintro.TrainingApp"

You should see a longer log output running by and end up with a folder `BinaryAndTrainer` containing a log file and a zip file.
Warnings about lingering threads at the end can be safely ignored, sometimes maven needs a short while to shut down the training app.
    
Open the project once in your IDE to see if you can see & edit everything.

To make sure everything is up to date, we recommand pulling the most recent state of this repo 
again shortly before the workshop. 
        
