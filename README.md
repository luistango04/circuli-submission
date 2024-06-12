Submission Scre Detection Model End to end

Solution

Software  - Objected Oriented Programming


 

Screw Detection: 
Solution: Ultimately The solution chosen was to train a convolutional Neural network ( YoloV8) with the received images. It showed positive signs with just a basic 3 hour CPU training and can be further improved to identify missing screws. In addition, given the other tasks that may be needed (such as putting in screws where holes are), it is likely that such a solution would be needed anyway to solve other tasks. Once the Screw was detected, a grayscale, blur , Hough transform was applied to find the circles in the detected image from YoloV8. 

Challenges Faced:
The following were attempted to see and they were either deemed to complicated of unfeasible due to the quality of the images. The solution was attempted first with the simpler solutions then I worked my way up in complexity until I arrived at a suitable outcome. 
•	Straight Hough Transform from Original data. (Filters on luminosity to remove glare followed be removal of background coloring. Showed great promise but showed to be very difficult to tune due to glare on screen as well as excessive background. Ended up using this a post processing task
•	Normalized Cross Correlation on a Preprocessed image – A traditional method that can be effective on things such as PCB boards. Given this problem was similar I had expected good results. However, even on the preprocessed gray scale image, substantial differences existed between the bolts. Thereby, reducing overall quality of the algorithm. 
•	MobileNet Trained Singleshot detector on grayscale image. – This is a similar Neural Network designed for IOT-Edge Computing. It performed admirable in detection but required a large amount of fine tuning. A success of 40% was achieved but not enough data was available to make the model properly work. This would be the recommended algorithm for this solution if I had more compute resources and time to preprocess the image and remove lens glare ( With physical changes to system).
•	YoloV8 Convolutional Neural Network – This was the network that was used for training and deployment of the app solution. It was very effective especially considering the poor lighting conditions, but, it is the equivalent of “bringing a bazooka to a knife fight”. Nevertheless, it works but great care needs to be taken around edge of the frame.
Screw X,Y,Z,Y,P,Y / Positioning
There were two solutions I conceived upon analysis. They can be simpliefied to whether you focus on the area around your screw or the screw itself. 
“Look at Screw” -Once the screw itself was identified in the image ( hough transform on the CNN model) those pixel positions served as a suitable mask for the pointcloud. It would be possible to draw a vector by taking the X,Y,Z mid planes (cutting the Pointcloud in half about each axis of the point cloud) and calculating Roll,Pitch,Yaw. From those planes. X,Y,Z would be the middle of the screw. The Solution would be effective near the center of the image but would likely collapse on the side where the optical view of the screw is distorted. This reduces the operational window of the camera ( not a good day at the office) for this task

“Look around the screw” (recommended) – Once these Limitations were identified I decided to look and see if I can capture the ring around the screw. It is generally understood that this area would be flat. Otherwise your screw would not have fun time doing its job. This proved more effective set of points for calculation of Roll pitch and yaw. Again the midpoint of the hough transform would serve was x,y,z. This is more effective generally speaking but has the risk of being distorted by obstacles. This is a known limitation that I originally hesitated. But then I realized I was making a robot un screw. In cases where this limitation exists, there could be an obstacle. In such cases this robot shouldn’t be unscrewing it anyway and trying to find a new path to screw. 

Implementation of The normal of the Screw. 
 To achieve this we took the cropped images from the open CV and calculated the normal to the surface of the plain clouds. KD tree was used to identify the normals and then a weight average of all of the normals was used. Improved performance took place due to the removal of the screw itself thereby giving you a better performance. The middle of the screw was then embedded with this normal and these were then transformed and rotated based on the transformation matrixes as provided in the JSON file.
Strengths:
•	It is very robust and can get to 100% performance within a certain sub region of the frame. If no circle is detected but the screws detected we still are able to identify the screw position. The normal planes half 4000 normal so it can be reduced even further if performance is needed.
•	The YoloV8n model is essentially bringing a bazooka to a knife fight and shows high performance and with further training can achieve near 100% results as well as detecting cases where the screw is on the gantry and we can remove those. We can also remove missing screws and you'll see in the results that missing screw are indeed not appearing.
•	Since the algorithm is plugged to a flask API it may be possible that many machines will be able to use the same algorithms by just deploying a thread provided the compute system has enough parallel cores to handle the task. What
Weaknesses
•	the decision to use YoloV8 what is the consequence of the glare on the image and may run too slowly for the ideal tasks. Attempting this solution I would try to use statistics such as the huge transform provided that I had more ability to handle the physical device, lenses, and lighting.
•	The parallel job libraries have sometimes shown to be difficult to close properly especially in debug mode.
•	Due to personal hardware limitations the algorithm runs and is trained on the CPU. This is the largest cause of the slowdown of the system. 



Results 
 
 
Figure Error! No text of specified style in document. 1 Results of Screw detection show great promise despite limited CPU testing
High Success rates in detecting Screws. Further Tuning needed for identification of the middle of the screw. It however shows great promise. 
 
Figure Error! No text of specified style in document. 2 3d Pointcoud projection and rendering show successful identificaiton in 3 dimensional space

