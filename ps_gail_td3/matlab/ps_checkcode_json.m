function report = ps_checkcode_json(path)
%ps_checkcode_json - Serialize MATLAB static findings for Python Engine
%   REPORT = ps_checkcode_json(PATH) returns a JSON array of findings.
%   See also checkcode

arguments
    path (1,:) char
end
report=jsonencode(checkcode(path,'-struct'));
end
