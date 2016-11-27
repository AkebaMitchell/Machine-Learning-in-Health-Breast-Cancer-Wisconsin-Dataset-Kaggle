%%%%% KNN on breast cancer - WIsconsin Dataset %%%%%%%%%%%%%
%% Loading data set and splitting into train and test 
clear all
load breast_cancer
summary(data);
X = table2array(data(:,3:32));   % columns as features 
y = table2array(data(:,2));   % label : benign or malignant
countpercent = tabulate(y) % displays proportion of classes as count and %

num_points = size(X,1);
split_point = round(num_points*0.7);
rng(10)
seq = randperm(num_points);
X_train = X(seq(1:split_point),:);
y_train = y(seq(1:split_point));
X_test = X(seq(split_point+1:end),:);
y_test = y(seq(split_point+1:end));


%% Creating function which computes kfoldloss using cross validated knn classification model  

neighbors = optimizableVariable('neighbors',[1,30],'Type','integer');
distance = optimizableVariable('distance',{'euclidean','minkowski', 'chebychev'},'Type','categorical');

n_samples = size(X_train, 1)
rng(10);
cv = cvpartition(n_samples,'Kfold',10);

% Loss function that contains kfoldloss and fitcknn
fun = @(x)kfoldLoss(fitcknn(X_train, y_train, 'CVPartition', cv,...
                    'NumNeighbors', x.neighbors,...
                    'Distance', char(x.distance), 'NSMethod', 'exhaustive'));
% Baysian Hyperparameter search
results = bayesopt(fun,[neighbors, distance],'Verbose',1,...
                   'AcquisitionFunctionName', 'expected-improvement-plus')

%%

% save the best parameters from the previous HP search 
neighbors = results.XAtMinObjective.neighbors;
distance = results.XAtMinObjective.distance;

rng(10);
% Apply 10-fold CV on optimal HPs
knn_tuned = fitcknn(X_train, y_train, 'CVPartition', cv,...
                    'NumNeighbors', neighbors,...
                    'Distance', char(distance), 'NSMethod', 'exhaustive');
                            
% compute average Accuracy score
error_knn = kfoldLoss(knn_tuned, 'mode', 'individual');
accuracy_knn = 100*(1 - mean(error_knn));
fprintf('\nAccuracy score of KNN is: %0.3f', accuracy_knn)

Predictedlabel = predict(knn_tuned.Trained{8},X_test); 

% compute  Accuracy score
[C,order] = confusionmat(y_test,Predictedlabel);
order 
precision =C(2,2)./(C(2,2)+C(1,2))

recall =  C(2,2)./(C(2,1)+C(2,2))


f1Score =  2*(precision.*recall)./(precision+recall)

