clear;

repo_root = fileparts(mfilename('fullpath'));
matlab_root = '/Users/bishwadeep/Downloads/tensors_decomp-2';

addpath(genpath(fullfile(matlab_root, 'utils')));

input_path = fullfile(repo_root, 'results', 'matlab_compare', 'contact_gsp_slice_input.mat');
output_path = fullfile(repo_root, 'results', 'matlab_compare', 'contact_gsp_slice_matlab_output.mat');

load(input_path, 'Z', 'a', 'b');

p = struct('verbosity', 0, 'maxit', 1000, 'tol', 1e-5, 'step_size', 0.5);
[A_hat, stat] = gsp_learn_graph_log_degrees(Z, a, b, p);

A_hat_normalized = A_hat;
max_value = max(A_hat(:));
if max_value > 0
    A_hat_normalized = A_hat / max_value;
end

save(output_path, 'A_hat', 'A_hat_normalized', 'stat', 'Z', 'a', 'b');
disp(['Saved MATLAB output to ', output_path]);
