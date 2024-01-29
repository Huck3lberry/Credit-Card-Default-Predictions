clc
clear all 

%% Initialize variables.
filename = 'defaultofcreditcardclients.csv';
delimiter = ',';
startRow = 3;

%% Format for each line of text:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Create output variable
defaultofcreditcardclients = table(dataArray{1:end-1}, 'VariableNames', {'ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','defaultpaymentnextmonth','reverse'});

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;


%% Data Wrangling to create a table for Descriptive Stats 
% Rename the imported data 
 df = defaultofcreditcardclients;
% Sort the data for the classess Default and Repayment 
df = sortrows(df,'defaultpaymentnextmonth','ascend');

% Select the data for the default cases 
datapay =df(1: 23364, 13:24);
agepay = df(1: 23364, 2);
limbalpay = df(1: 23364, 6);

% Select the data for the Pay cases 
datadef =df(23365:30000, 13:24);
agedef = df(23365:30000, 2);
limbaldef = df(23365:30000,6);

%Concatenate the separate datasets for the classes 
dfpay = [agepay,limbalpay, datapay];
dfdef = [agedef, limbaldef, datadef];

%% Descriptive Statistics Table

% Create a table with the stats for the Pay data (Class2)
t2 = varfun(@(x) [mean(x);std(x);median(x); max(x); min(x); skewness(x); kurtosis(x);jbtest(x);],dfpay);

% Names for the rows 
t2.Properties.RowNames = {'Mean Pay' 'Std Pay' 'Median Pay' 'Max Pay' 'Min Pay' 'Skewness Pay' 'Kurtosis Pay' 'JB Pay'};

% Fix the columns names 
t2.Properties.VariableNames = extractAfter(t2.Properties.VariableNames,'Fun_');

% convert the rows to vars to save space for the poster 
t2 = rows2vars(t2);

% Create a table for the Default data (Class1) (same procedure as above)
t1 = varfun(@(x) [mean(x);std(x); median(x); max(x); min(x); skewness(x); kurtosis(x);jbtest(x);],dfdef);

% Row names 
t1.Properties.RowNames = {'Mean Def' 'Std Def' 'Median Def' 'Max Def' 'Min Def' 'Skew Def' 'Kurtosis Def' 'JB Def'};

% Format the columns names 
t1.Properties.VariableNames = extractAfter(t1.Properties.VariableNames,'Fun_');

% Convert the rows to vars to save space 
t1 = rows2vars(t1);

% Concatenate the two tables 
t1.Properties.VariableNames{1} = 'OVN';
t3 = [t1,t2];


%% Data Wrangling ANN
data= defaultofcreditcardclients(:,2:24);
% labels = defaultofcreditcardclients(:,25);
% labels1 = defaultofcreditcardclients(:,26);
% inputs = [labels,labels1];
inputs = defaultofcreditcardclients(:,25:26);

%% ANN
%Convert tables to array 
xtrain = table2array(data);
xinputs = table2array(inputs);

% create the varibles for the network 
x = xtrain';
t = xinputs';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 2;
net = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Set the main parameters of the model 
net.trainParam.lr = 0.01; % Learning Rate 
net.trainParam.mc = 0; % Momentum 

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance using cross entropy
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%  figure, plotperform(tr)
%  figure, plottrainstate(tr)
%  figure, ploterrhist(e)
figure, plotconfusion(t,y)
%  figure, plotroc(t,y)

%Confusion matrices filled in with the plot values 
C_net = [2209 434; 
        110 247];
C_net_val = [2202 403; 
            132 262]; 

%Compute F1 score
F1_net = 2*C_net(1,1)/(2*C_net(1,1)+C_net(2,1)+C_net(1,2));
F1_net_val = 2*C_net_val(1,1)/(2*C_net_val(1,1)+C_net_val(2,1)+C_net_val(1,2));

%Calculation of the Classification Errors for NN
CE_net = (C_net(1,2)+C_net(2,1))/(C_net(1,2)+C_net(2,1)+C_net(2,2)+C_net(1,1));
CE_net_val = (C_net_val(1,2)+C_net_val(2,1))/(C_net_val(1,2)+C_net_val(2,1)+C_net_val(2,2)+C_net_val(1,1));

%Obtain Accuracy
A_net = 1 - CE_net; 
A_net_val = 1 - CE_net_val; 

%% SVM Data Wrangling 

% Shuffle the data (as it is originally ordered)
df = table2array(df);
df1=df(randsample(1:length(df),length(df)),:);

% Create a train, validation and test sets 
labels = df1(:,25);
trainSVM = df1(1:24000,2:24);
valSVM = df1(24000:27000,2:24);
testSVM = df1(27000:30000,2:24);
lab_trainSVM = labels(1:24000,:);
lab_valSVM = labels(24000:27000,:);
lab_testSVM = labels(27000:30000,:);


% SVM

% Fit SVM - linear, polynomial, rbf 
mdlSVM = fitcsvm(trainSVM,lab_trainSVM,'Standardize',true,'KernelFunction','polynomial');

% Predict on the Validation set 
val_hat = mdlSVM.predict(valSVM);

%Predict on the Test set 
test_hat = mdlSVM.predict(testSVM);

% Confusion Matrices for the test and val data 
C_test = confusionmat(lab_testSVM,test_hat);
C_val = confusionmat(lab_valSVM, val_hat);

% Support Vectors 
% sv = mdlSVM.SupportVectors;
% figure;
% gscatter(trainSVM(:,2),trainSVM(:,3),lab_trainSVM)
% hold on
% plot(sv(:,1),sv(:,1),'ko','MarkerSize',10)
% legend('Default','Payment','Support Vector')
% hold off

% Compute F1 for the test and val sets
F1_test = 2*C_test(1,1)/(2*C_test(1,1)+C_test(2,1)+C_test(1,2));
F1_val = 2*C_val(1,1)/(2*C_val(1,1)+C_val(2,1)+C_val(1,2));

% Calculation of the Classification Errors
CE_test = (C_test(1,2)+C_test(2,1))/(C_test(1,2)+C_test(2,1)+C_test(2,2)+C_test(1,1));
CE_val = (C_val(1,2)+C_val(2,1))/(C_val(1,2)+C_val(2,1)+C_val(2,2)+C_val(1,1));

% Accuracy   
A_test = 1 - CE_test;
A_val = 1 - CE_val; 

%% Plot a correlation matrix 
pl = df(:,2);
pl1 = df(:,13:24);
cpl = [pl pl1];

figure;
corrplot(cpl)


