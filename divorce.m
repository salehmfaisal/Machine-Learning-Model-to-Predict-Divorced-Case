% Loading Divorce data set
%
load divorce.csv
%
% Loading class or lebels of 
load class.csv
%
%Building data Matrix
%
X=divorce;
y=class;
%Splitting data set into training and testing data into (80:20)
rand_num = randperm(size(X,1));
X_train = X(rand_num(1:round(0.8*length(rand_num))),:);
y_train = y(rand_num(1:round(0.8*length(rand_num))),:);
X_test = X(rand_num(round(0.8*length(rand_num))+1:end),:);
y_test = y(rand_num(round(0.8*length(rand_num))+1:end),:);

%Cross Validation Partition
%
c=cvpartition(y_train,'k',10);
%Selecting best features
%
options=statset('Display', 'iter');
classfunction=@(X_train, y_train, X_test, y_test)loss(fitcecoc(X_train,y_train),X_test,y_test);
[features,history]=sequentialfs(classfunction, X_train, y_train, 'cv',c,'options',options,'nfeatures',2);
%
%best X_train and X_test
best_X_train=X_train(:,features);
best_X_test=X_test(:,features);
%
%Lets run a model 
%
linear_model=linearclassifier(best_X_train, y_train);
%
%Prediction
%
predictions = best_X_test*linear_model.X'*(linear_model.a.*linear_model.y) - linear_model.b > 0;
%
%Checking Accuracy
%
accuracy = mean(predictions==(y_test));
%
%nonlinear approach
%
nonlinear_model = nonlinearclassifier(best_X_train, y_train,.165,0);
%
%prediction for nonlinear model
%
K=best_X_test*nonlinear_model.X';
nonlinear_predictions = K*(nonlinear_model.a.*nonlinear_model.y)-nonlinear_model.b>0;
%
%Accuracy for non linear model
%
nonlinear_accuracy = mean(nonlinear_predictions==(y_test));
%
%MATLAB built-in function
%
model=fitcsvm(best_X_train, y_train,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      'expected-improvement-plus','ShowPlots',true))
 %Accuracy
 % 
accuracy=sum(predict(model,best_X_test)==y_test)/length(y_test)*100;
  %
  %Plots for optimal hyperplane
  %
   figure;
scatter = gscatter(best_X_train(:,1),best_X_train(:,2),y_train);
hold on;
hyperplane_support_vectors=plot(model.SupportVectors(:,1),model.SupportVectors(:,2),'ko','markersize',8);
%test set preparation
gscatter(best_X_test(:,1),best_X_test(:,2),y_test,'rb','xx')

% decision plane
Xlims = get(gca,'xlim');
Ylims = get(gca,'ylim');
[xi,yi] = meshgrid([Xlims(1):0.01:Xlims(2)],[Ylims(1):0.01:Ylims(2)]);
grid = [xi(:), yi(:)];
pred_mesh = predict(model, grid);
redcolor = [1, 0.8, 0.8];
bluecolor = [0.8, 0.8, 1];
position = find(pred_mesh == 1);
Hyperplane1 = plot(grid(position,1), grid(position,2),'s','color',redcolor,'Markersize',5,'MarkerEdgeColor',redcolor,'MarkerFaceColor',redcolor);
position = find(pred_mesh == 2);
Hyperplane2 = plot(grid(position,1), grid(position,2),'s','color',bluecolor,'Markersize',5,'MarkerEdgeColor',bluecolor,'MarkerFaceColor',bluecolor);
uistack(Hyperplane1,'bottom');
uistack(Hyperplane2,'bottom');
legend([scatter;hyperplane_support_vectors],{'Atrr6','Atrr11','support vectors'})