% Load dataset
load('smiles.mat'); 
% Training
A = [];
for k = 1:size(train_data, 3)
    p = reshape(train_data(:,:,k), 1, 576);
    A = [A; p, 1]; 
end
w = pinv(A) * double(smile_flag_train);  % Explicitly convert to double
% Testing
B = [];
for k = 1:size(test_data, 3)
    p = reshape(test_data(:,:,k), 1, 576);
    B = [B; p, 1];
end
scores = B * w;
% Classification
threshold = 0.587;  % Optimal threshold (updated via ROC)
predictions = scores > threshold;
% Evaluation
test_labels = logical(smile_flag_test(:));  % Ensure column vector
predictions = logical(predictions(:));     % Ensure column vector
% Confusion Matrix
C = confusionmat(test_labels, predictions);
disp('Confusion Matrix:');
disp(C);
% Accuracy (fixed)
accuracy = mean(predictions == test_labels) * 100;
disp(['Accuracy: ', num2str(accuracy), '%']);
% ROC Curve
[X, Y, T] = perfcurve(test_labels, scores, true);
figure;
plot(X, Y, 'b-', 'LineWidth', 1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Linear Regression (24x24)');
grid on;
% Optimal Threshold (Youden's J)
[~, idx] = max(Y - X);
optimal_threshold = T(idx);
hold on;
plot(X(idx), Y(idx), 'ro', 'MarkerSize', 8);
text(X(idx)+0.05, Y(idx)-0.05, ...
    sprintf('Optimal Threshold = %.3f', optimal_threshold), ...
    'FontSize', 10);
hold off;