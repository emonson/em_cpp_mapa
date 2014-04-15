% A test function for test_rand_uniform_on_hypersphere.m

%subplot(1,3,1)
figure(1)
X = zeros(1,100);
tic
X = rand_uniform_on_hypersphere(1,100);
toc
plot(X(1,:),X(1,:), 'bo')

figure(2)
%subplot(1,3,2)
X = ones(2,400);
tic
X = rand_uniform_on_hypersphere(2,400);
toc
plot(X(1,:),X(2,:),'b.');

figure(3)
%subplot(1,3,3)
X = zeros(3,4000);
tic
X = rand_uniform_on_hypersphere(3,4000);
toc
plot3(X(1,:),X(2,:),X(3,:),'b.');

% % Figure out which kind of random number is easier to generate.
% pointCount = 2000;
% X = zeros(2000);
% tic
% X = rand(2000);
% toc
% tic
% X = randn(2000);
% toc
% % Uniform generation is actually a little slower than normal distribution.