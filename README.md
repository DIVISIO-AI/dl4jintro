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

    git clone git@github.com:DIVSIO/dl4jintro.git
    
Build the project once to do download all dependencies, sources & javadoc. DL4J has a lot of dependencies, 
so it is better if you downloaded them before the workshop.

    mvn clean package -DdownloadSources=true -DdownloadJavadocs=true
    
If you want, you can run the application once like so: 

    mvn exec:java -Dexec.mainClass="divisio.dl4jintro.TrainingApp" -Dexec.args="-wf training_1bit_and -e 100"

You should see a longer log output running by and end up with a folder `training_1bit_and` containing a log file and a zip file.
Any SLF4J warnings at the beginning and warnings about lingering threads at the end can be safely ignored.
    
Open the project once in your IDE to see if you can see & edit everything.

To make sure everything is up to date, we recommand pulling the most recent state of this repo 
again shortly before the workshop. 
        
