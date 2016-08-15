PAIRS = 10;
ITERS = 8;
CORES=1;
PATCH_W = 3;

runtimes = zeros(1, 5);
l2dists = zeros(1, 5);
matchcounter = 0;
total = tic;
for i = 1:2:PAIRS*2
    matchcounter = matchcounter + 1;
    disp(['Matching image pair: ', num2str(matchcounter)]);
    filenameA = strcat('E:\STACK\Bachelor Thesis\Vidpairs_Dataset\small\vidpair',num2str(i),'.jpg');
    filenameB = strcat('E:\STACK\Bachelor Thesis\Vidpairs_Dataset\small\vidpair',num2str(i+1),'.jpg');
    A=imread(filenameA);
    B=imread(filenameB);
    B=B(1:end-1,1:end-1,:);
    totalimage = tic;
    for j = 1:ITERS
        iter = tic;
        ann0 = nnmex(A, B, 'cputiled', PATCH_W, j, [], [], [], [], CORES);
        time = toc(iter);
        disp(['NN A -> B time: ', num2str(time), ' sec with ', num2str(j), ' iterations']);
        runtimes(matchcounter,j) = time;
        l2dists(matchcounter,j) = mean2(sqrt(double(ann0(1:end-PATCH_W,1:end-PATCH_W,3))));
    end
    timeimage = toc(totalimage);
    disp(['Image took total time of: ', num2str(timeimage), ' sec']);
end
disp(['Total runtime: ', num2str(toc(total)), ' sec']);

% Used when plotting final results. Also plots results from our approach
% which is expected to be stored as test.mat
% t = load('C:\Users\Jeftha\PycharmProjects\bachelorthesis\output\test.mat');
% x = mean(runtimes, 1);
% y = mean(l2dists, 1);
% v = t.averages;
% u = t.times;
% p = plot(x, y, u, v);
% xlabel('seconds');
% ylabel('L2 dist');
% p(1).Marker = 'square';
% p(2).Marker = 'square';
% title('10 image pairs, patch size 3, image size 500*208 pixels')
% legend('PatchMatch', 'k-d tree + PCA')
% legend('show')