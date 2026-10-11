function record = s6_load_record(mat_file)
%S6_LOAD_RECORD Load a table-2 trajectory record and standardize fields.
%   Adapted from plot_table2_matlab/load_table2_records.m (read-only).

data = load(mat_file);
if ~isfield(data, 'record') || ~isstruct(data.record)
    error('s6_load_record:BadFile', 'no record struct in %s', mat_file);
end
record = data.record;

required = {'t', 'pos_M', 'pos_D1', 'pos_D2', 'a_M'};
for k = 1:numel(required)
    if ~isfield(record, required{k})
        error('s6_load_record:MissingField', ...
            'record missing field %s: %s', required{k}, mat_file);
    end
end

record.t = double(record.t(:).');
record.pos_M = as_3_by_n(record.pos_M);
record.pos_D1 = as_3_by_n(record.pos_D1);
record.pos_D2 = as_3_by_n(record.pos_D2);

n = min([numel(record.t), size(record.pos_M, 2), ...
    size(record.pos_D1, 2), size(record.pos_D2, 2)]);
if n < 2
    error('s6_load_record:TooShort', 'too few points: %s', mat_file);
end
record.t = record.t(1:n);
record.pos_M = record.pos_M(:, 1:n);
record.pos_D1 = record.pos_D1(:, 1:n);
record.pos_D2 = record.pos_D2(:, 1:n);

a_M = double(record.a_M);
if size(a_M, 1) ~= 2 && size(a_M, 2) == 2
    a_M = a_M.';
end
record.a_M = fit_columns(a_M(1:2, :), n, NaN);

if isfield(record, 'T') && numel(record.T) >= 3
    target = double(record.T(1:3));
    target = target(:);
else
    target = zeros(3, 1);
    record.T = target.';
end

record.r_MD1 = get_or_derive(record, 'r_MD1', record.pos_M, record.pos_D1, n);
record.r_MD2 = get_or_derive(record, 'r_MD2', record.pos_M, record.pos_D2, n);
record.r_MT = get_or_derive(record, 'r_MT', record.pos_M, ...
    repmat(target, 1, n), n);
record.a_n = hypot(record.a_M(1, :), record.a_M(2, :));
end


function value = as_3_by_n(value)
value = double(value);
if size(value, 1) ~= 3 && size(value, 2) == 3
    value = value.';
end
end


function value = get_or_derive(record, field_name, p1, p2, n)
if isfield(record, field_name) && ~isempty(record.(field_name))
    value = double(record.(field_name));
    value = value(:).';
    value = fit_columns(value, n, NaN);
else
    value = sqrt(sum((p1 - p2).^2, 1));
end
bad = ~isfinite(value);
if any(bad)
    derived = sqrt(sum((p1 - p2).^2, 1));
    value(bad) = derived(bad);
end
end


function value = fit_columns(value, n, fill_value)
if size(value, 2) >= n
    value = value(:, 1:n);
else
    value(:, end + 1:n) = fill_value;
end
end
