d = load('FCWDemoMonoCameraSensor.mat', 'sensor');

detector = vehicleDetectorACF('full-view');
vehicleWidth = [1.5, 2.5];

detector = configureDetectorMonoCamera(detector, d.sensor, vehicleWidth);

[tracker, positionSelector] = setupTracker();

videoFile   = '05_highway_lanechange_25s.mp4';
videoReader = VideoReader(videoFile);
videoPlayer = vision.DeployableVideoPlayer();

currentStep = 0;
snapshot = [];
snapTimeStamp = 120;
cont = hasFrame(videoReader);
while cont
    currentStep = currentStep + 1;
        
    frame = readFrame(videoReader);
    
    detections = detectObjects(detector, frame, currentStep);
       
    confirmedTracks = updateTracks(tracker, detections, currentStep);
    
    confirmedTracks = removeNoisyTracks(confirmedTracks, positionSelector, d.sensor.Intrinsics.ImageSize);
    
    frameWithAnnotations = insertTrackBoxes(frame, confirmedTracks, positionSelector, d.sensor);

    videoPlayer(frameWithAnnotations);  
    
    if currentStep == snapTimeStamp
        snapshot = frameWithAnnotations;
    end   
    
    cont = hasFrame(videoReader) && isOpen(videoPlayer);
end

if ~isempty(snapshot)
    figure
    imshow(snapshot)
end

function [tracker, positionSelector] = setupTracker()
    % Create the tracker object.
    tracker = multiObjectTracker('FilterInitializationFcn', @initBboxFilter, ...
        'AssignmentThreshold', 50, ...
        'NumCoastingUpdates', 5, ... 
        'ConfirmationParameters', [3 5]);

    % The State vector is: [x; vx; y; vy; w; vw; h; vh]
    % [x;y;w;h] = positionSelector * State
    positionSelector = [1 0 0 0 0 0 0 0; ...
                        0 0 1 0 0 0 0 0; ...
                        0 0 0 0 1 0 0 0; ...
                        0 0 0 0 0 0 1 0]; 
end
function filter = initBboxFilter(Detection)
    dt = 1;
    cvel =[1 dt; 0 1];
    A = blkdiag(cvel, cvel, cvel, cvel);
 
    H = [1 0 0 0 0 0 0 0; ...
         0 0 1 0 0 0 0 0; ...
         0 0 0 0 1 0 0 0; ...
         0 0 0 0 0 0 1 0];
 
    state = [Detection.Measurement(1); ...
             0; ...
             Detection.Measurement(2); ...
             0; ...
             Detection.Measurement(3); ...
             0; ...
             Detection.Measurement(4); ...
             0];
 
    L = 100; 
    stateCov = diag([Detection.MeasurementNoise(1,1), ...
                     L, ...
                     Detection.MeasurementNoise(2,2), ...
                     L, ...
                     Detection.MeasurementNoise(3,3), ...
                     L, ...
                     Detection.MeasurementNoise(4,4), ...
                     L]);
 
    filter = trackingKF(...
        'StateTransitionModel', A, ...
        'MeasurementModel', H, ...
        'State', state, ...
        'StateCovariance', stateCov, ... 
        'MeasurementNoise', Detection.MeasurementNoise, ...
        'ProcessNoise', Q);
end
function detections = detectObjects(detector, frame, frameCount)
    bboxes = detect(detector, frame);
    
    L = 100;
    measurementNoise = [L 0  0  0; ...
                        0 L  0  0; ...
                        0 0 L/2 0; ...
                        0 0  0 L/2];
                    
    numDetections = size(bboxes, 1);
    detections = cell(numDetections, 1);                      
    for i = 1:numDetections
        detections{i} = objectDetection(frameCount, bboxes(i, :), ...
            'MeasurementNoise', measurementNoise);
    end
end
function tracks = removeNoisyTracks(tracks, positionSelector, imageSize)

    if isempty(tracks)
        return
    end
    
    positions = getTrackPositions(tracks, positionSelector);
    invalid = ( positions(:, 1) < 1 | ...
                positions(:, 1) + positions(:, 3) > imageSize(2) | ...
                positions(:, 3) <= 20 | ...
                positions(:, 4) <= 20 );
    tracks(invalid) = [];
end
function I = insertTrackBoxes(I, tracks, positionSelector, sensor)

    if isempty(tracks)
        return
    end

    labels = cell(numel(tracks), 1);
    bboxes = getTrackPositions(tracks, positionSelector);

    for i = 1:numel(tracks)        
        box = bboxes(i, :);
        
        xyVehicle = imageToVehicle(sensor, [box(1)+box(3)/2, box(2)+box(4)]);
        
        labels{i} = sprintf('x=%.1f,y=%.1f',xyVehicle(1),xyVehicle(2));
    end
    
    I = insertObjectAnnotation(I, 'rectangle', bboxes, labels, 'Color', 'yellow', ...
        'FontSize', 10, 'TextBoxOpacity', .8, 'LineWidth', 2);
end
