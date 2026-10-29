#!/usr/bin/env perl
use strict;
use warnings;
use MIME::Base64 qw(encode_base64);

my $svg_path = shift @ARGV or die "usage: $0 path/to/figure.svg\n";

open my $in, '<:raw', $svg_path or die "cannot read $svg_path: $!\n";
local $/;
my $svg = <$in>;
close $in;

$svg =~ s{xlink:href="file://([^"\n]+\.png)"}{
    my $image_path = $1;
    open my $image, '<:raw', $image_path
        or die "cannot read embedded image $image_path: $!\n";
    local $/;
    my $bytes = <$image>;
    close $image;
    'xlink:href="data:image/png;base64,' . encode_base64($bytes, '') . '"';
}ge;

my $tmp_path = "$svg_path.tmp";
open my $out, '>:raw', $tmp_path or die "cannot write $tmp_path: $!\n";
print {$out} $svg;
close $out or die "cannot close $tmp_path: $!\n";
rename $tmp_path, $svg_path or die "cannot replace $svg_path: $!\n";

