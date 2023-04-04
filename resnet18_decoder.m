function dlnet = resnet18_decoder(projectionSize, numLatentChannels)
%RESNET18_ENCODER Summary of this function goes here
%   Detailed explanation goes here
lgraph = layerGraph();

tempLayers = [
    featureInputLayer(256,"Name","featureinput")
    fullyConnectedLayer(512,"Name","fc")
    projectAndReshapeLayer(projectionSize,2*numLatentChannels,"Name","depthToSpace")
    resize3dLayer("Name","resize3d-output-size_6","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 512])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","x_decoder_layer4__19","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer4__16")
    convolution2dLayer([3 3],512,"Name","x_decoder_layer4__18","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer4__15")
    reluLayer("Name","x_decoder_layer4__17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","x_decoder_layer4__26","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer4__21")
    resize3dLayer("Name","resize3d-output-size_4","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 512])
    convolution2dLayer([3 3],256,"Name","x_decoder_layer4__25","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_5","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 512])
    convolution2dLayer([3 3],256,"Name","x_decoder_layer4__29","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer4__20")
    reluLayer("Name","x_decoder_layer4__22")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","x_decoder_layer3__19","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer3__16")
    convolution2dLayer([3 3],256,"Name","x_decoder_layer3__18","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer3__15")
    reluLayer("Name","x_decoder_layer3__17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","x_decoder_layer3__26","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer3__21")
    resize3dLayer("Name","resize3d-output-size_3","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 256])
    convolution2dLayer([3 3],128,"Name","x_decoder_layer3__25","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 256])
    convolution2dLayer([3 3],128,"Name","x_decoder_layer3__29","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer3__20")
    reluLayer("Name","x_decoder_layer3__22")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","x_decoder_layer2__19","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer2__16")
    convolution2dLayer([3 3],128,"Name","x_decoder_layer2__18","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer2__15")
    reluLayer("Name","x_decoder_layer2__17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","x_decoder_layer2__26","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer2__21")
    resize3dLayer("Name","resize3d-output-size_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 128])
    convolution2dLayer([3 3],64,"Name","x_decoder_layer2__25","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 128])
    convolution2dLayer([3 3],64,"Name","x_decoder_layer2__29","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer2__20")
    reluLayer("Name","x_decoder_layer2__22")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","x_decoder_layer1__14","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer1__11")
    convolution2dLayer([3 3],64,"Name","x_decoder_layer1__13","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer1__10")
    reluLayer("Name","x_decoder_layer1__12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","x_decoder_layer1__19","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    reluLayer("Name","x_decoder_layer1__16")
    convolution2dLayer([3 3],64,"Name","x_decoder_layer1__18","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","x_decoder_layer1__15")
    reluLayer("Name","x_decoder_layer1__17")
    convolution2dLayer([3 3],3,"Name","conv","Padding","same")
    tanhLayer("Name","tanh")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"resize3d-output-size_6","x_decoder_layer4__19");
lgraph = connectLayers(lgraph,"resize3d-output-size_6","x_decoder_layer4__15/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer4__18","x_decoder_layer4__15/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer4__17","x_decoder_layer4__26");
lgraph = connectLayers(lgraph,"x_decoder_layer4__17","resize3d-output-size_5");
lgraph = connectLayers(lgraph,"x_decoder_layer4__25","x_decoder_layer4__20/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer4__29","x_decoder_layer4__20/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer4__22","x_decoder_layer3__19");
lgraph = connectLayers(lgraph,"x_decoder_layer4__22","x_decoder_layer3__15/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer3__18","x_decoder_layer3__15/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer3__17","x_decoder_layer3__26");
lgraph = connectLayers(lgraph,"x_decoder_layer3__17","resize3d-output-size_2");
lgraph = connectLayers(lgraph,"x_decoder_layer3__25","x_decoder_layer3__20/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer3__29","x_decoder_layer3__20/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer3__22","x_decoder_layer2__19");
lgraph = connectLayers(lgraph,"x_decoder_layer3__22","x_decoder_layer2__15/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer2__18","x_decoder_layer2__15/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer2__17","x_decoder_layer2__26");
lgraph = connectLayers(lgraph,"x_decoder_layer2__17","resize3d-output-size");
lgraph = connectLayers(lgraph,"x_decoder_layer2__25","x_decoder_layer2__20/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer2__29","x_decoder_layer2__20/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer2__22","x_decoder_layer1__14");
lgraph = connectLayers(lgraph,"x_decoder_layer2__22","x_decoder_layer1__10/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer1__13","x_decoder_layer1__10/in1");
lgraph = connectLayers(lgraph,"x_decoder_layer1__12","x_decoder_layer1__19");
lgraph = connectLayers(lgraph,"x_decoder_layer1__12","x_decoder_layer1__15/in2");
lgraph = connectLayers(lgraph,"x_decoder_layer1__18","x_decoder_layer1__15/in1");

dlnet = lgraph;
end

