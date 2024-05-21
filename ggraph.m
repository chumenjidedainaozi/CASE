num_drugs = 100; 
num_features = 2;
features = gdata(num_drugs, num_features); 

num_clusters = 4;
idx = kmeans(features, num_clusters);

colors = jet(num_clusters);
figure;
hold on;
for i = 1:num_clusters
    cluster_indices = find(idx == i);
    scatter(features(cluster_indices, 1), features(cluster_indices, 2), 50, colors(i, :), 'filled');
end
hold off;
title('Drug characteristic clustering ');
xlabel('T-SNE1');
ylabel('T-SNE2');
legend('Fibromyalgia', 'Pancreatitis', 'Uterine polyp', 'Viral meningitis');

side_effects = zeros(num_drugs, 1);
num_side_effects = round(num_drugs * 0.2); 
side_effects(randperm(num_drugs, num_side_effects)) = 1;

figure;
hold on;
for i = 1:num_clusters
    cluster_indices = find(idx == i);
    scatter(features(cluster_indices, 1), features(cluster_indices, 2), 50, colors(i, :), 'filled');
end
scatter(features(side_effects == 1, 1), features(side_effects == 1, 2), 'rx');
hold off;
title('Drug characteristic clustering and Side effect');
xlabel('T-SNE1');
ylabel('T-SNE2');
legend('Fibromyalgia', 'Pancreatitis', 'Uterine polyp', 'Viral meningitis', 'Side effect');


function data = gdata(num_rows, num_cols)
    data = randn(num_rows, num_cols);
end