
%%%%%%%%%%%% Weighted Principal Component Analysis %%%%%%%%%%%
clear all
load breast_cancer
%load cancercleaned
categories = data.Properties.VariableNames;
categories = categories(3:32)
%summary(data);
X = table2array(data(:,3:32));   % columns as features 
y = table2array(data(:,2));   % label : benign or malignant
countpercent = tabulate(y) % displays proportion of classes as count and %
figure()
boxplot(X,'orientation','horizontal','labels',categories)
coeff = pca(X(:,1:30),'Rows','all');
C = corr(X,X);

% Scaling 

w = 1./var(X);
[wcoeff,score,latent,tsquared,explained] = pca(X,...
'VariableWeights',w);
%[wcoeff,score,latent,tsquared,explained] = pca(X,...
%'VariableWeights','variance');

c3 = wcoeff(:,1:3)
coefforth = inv(diag(std(X)))*wcoeff;
I = coefforth'*coefforth;
I(1:3,1:3)

cscores = zscore(X)*coefforth;

figure()
plot(score(:,1),score(:,2),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')

latent
explained

figure()
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

%Visualising results 
biplot(coefforth(:,1:2),'scores',score(:,1:2))%'varlabels',categories);
axis([-.26 0.6 -.51 .51]);

%Visualising results in 3d
figure()
biplot(coefforth(:,1:3),'scores',score(:,1:3));
axis([-.26 0.8 -.51 .51 -.61 .81]);
view([30 40]);

%% Now running kNN on the projected data 
X = score;                            
num_points = size(X,1);
split_point = round(num_points*0.7);
rng(10)
seq = randperm(num_points);
X_train = X(seq(1:split_point),:);
TrainingLabel = y(seq(1:split_point));
X_test = X(seq(split_point+1:end),:);
Testinglabel = y(seq(split_point+1:end));

neighbors = optimizableVariable('neighbors',[1,30],'Type','integer');
distance = optimizableVariable('distance',{'euclidean','minkowski', 'chebychev'},'Type','categorical');

n_samples = size(X_train, 1)
rng(10);
cv = cvpartition(n_samples,'Kfold',10);

% Loss function that contains kfoldloss and fitcknn
fun = @(x)kfoldLoss(fitcknn(X_train, TrainingLabel, 'CVPartition', cv,...
                    'NumNeighbors', x.neighbors,...
                    'Distance', char(x.distance), 'NSMethod', 'exhaustive'));
% Baysian Hyperparameter search
results = bayesopt(fun,[neighbors, distance],'Verbose',1,...
                   'AcquisitionFunctionName', 'expected-improvement-plus')

% save the best parameters from the previous HP search 
neighbors = results.XAtMinObjective.neighbors;
distance = results.XAtMinObjective.distance;

rng(10);
% Apply 10-fold CV on optimal HPs
knn_tuned = fitcknn(X_train, TrainingLabel, 'CVPartition', cv,...
                    'NumNeighbors', neighbors,...
                    'Distance', char(distance), 'NSMethod', 'exhaustive');
                
Predictedlabel = predict(knn_tuned.Trained{8},X_test);     

% compute  Accuracy score
[C,order] = confusionmat(Testinglabel,Predictedlabel);
order 
precision =C(2,2)./(C(2,2)+C(1,2))

recall =  C(2,2)./(C(2,1)+C(2,2))


f1Score =  2*(precision.*recall)./(precision+recall)
                             