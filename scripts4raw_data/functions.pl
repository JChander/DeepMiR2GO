#!/ifs/share/bin/perl/bin/perl -w

use lib qw(/ifs/home/test/lib);
use warnings;
use strict;

#&deal_RePeate();
#&merge_multi_GO();
&line2row();

sub deal_RePeate(){
	open(INFILE,"/ifs/data1/wangjiacheng/2step2/goa/GOA_ensg_nonIEA_201604.txt")||die "this file not exists\n";
	open(OUTFILE,">/ifs/data1/wangjiacheng/2step2/goa/GOA_ensg_nonIEA_201604_noRepeat.txt")||die "Cannot open the newfile:$!\n";

	my %hash;
	my @input;
	my @array;
	while(<INFILE>){
		chomp;
		@input = split"\t";
		#$tmp[1] =~ tr/PS//d;
		my $temp = $input[0]."\t".$input[1]."\t".$input[2];
	#print $temp,"\n";
		if(!exists($hash{$temp})){
			$hash{$temp} = 1;
		}
	}

	@array=keys%hash;
	my $lines=@array;
	for(my $i=0;$i<$lines;$i++){
		print OUTFILE $array[$i],"\n";
	}
	close(OUTFILE);
	close(INFILE);
}

sub merge_multi_GO(){
	open(INFILE,"/ifs/data1/wangjiacheng/2step2/goa/GOs_mf.txt")||die "this file not exists\n";
	open(OUTFILE,">/ifs/data1/wangjiacheng/2step2/goa/GOs_mf_ros.txt")||die "Cannot open the newfile:$!\n";

	my $SUP = "IEA";
	my %hash;
	my @input;
	while(<INFILE>){
		chomp;
		@input = split"\t";
		next if $input[2] eq $SUP;
		#my $temp = $input[0]."\t".$input[1];
		if(!exists($hash{$input[0]})){
			$hash{$input[0]} = $input[1];
		}else{
			$hash{$input[0]} =$hash{$input[0]}.",".$input[1];
		}
	}

	my @array = keys%hash;
	my $lines = @array;
	for(my $i = 0; $i < $lines; $i++){
		print OUTFILE $array[$i],"\t",$hash{$array[$i]},"\n";
	}
	close(OUTFILE);
	close(INFILE);
}

sub line2row(){
	open(INFILE,"/ifs/data1/wangjiacheng/2step2/goa/hsa-18a-5p.txt")||die "this file not exists\n";
	open(OUTFILE,">/ifs/data1/wangjiacheng/2step2/goa/hsa-18a-5p_row.txt")||die "Cannot open the newfile:$!\n";

	my @input;
	while(<INFILE>){
		chomp;
		@input = split" ";
		my $acount = @input;
		for(my $i = 0; $i < $acount; $i++){
			print OUTFILE $input[$i],"\n";
		}
	}
	close(OUTFILE);
	close(INFILE);
}