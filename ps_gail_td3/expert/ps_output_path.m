function path = ps_output_path(path)
%ps_output_path - Validate an artifact destination within this branch
%   PATH = ps_output_path(PATH) resolves and checks the output path.
%   See also generate_radau_library

arguments
    path (1,:) char
end
branch=fileparts(fileparts(mfilename('fullpath')));
path=char(java.io.File(path).getCanonicalPath());
assert(startsWith(lower(path),[lower(branch),filesep]), ...
    'Output must remain inside ps_gail_td3');
parent=fileparts(path); if ~isfolder(parent); mkdir(parent); end
end
