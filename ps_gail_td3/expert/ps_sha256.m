function sha = ps_sha256(path)
%ps_sha256 - Calculate the SHA256 fingerprint of a local file
%   SHA = ps_sha256(PATH) returns a lowercase hexadecimal digest.
%   See also generate_radau_library

arguments
    path (1,:) char
end
fid=fopen(path,'rb'); assert(fid>=0); cleanup=onCleanup(@() fclose(fid));
md=java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    md.update(fread(fid,1048576,'*uint8'));
end
sha=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
