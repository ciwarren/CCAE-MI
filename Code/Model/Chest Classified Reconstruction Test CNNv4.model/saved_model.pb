??1
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8ɛ-
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_9/gamma
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_9/beta
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:0*
dtype0
?
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_9/moving_mean
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:0*
dtype0
?
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_9/moving_variance
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:0@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_5/kernel
?
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_6/kernel
?
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
: *
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0 **
shared_nameconv2d_transpose_7/kernel
?
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*&
_output_shapes
:0 *
dtype0
?
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:0*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:0*
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@0**
shared_nameconv2d_transpose_8/kernel
?
-conv2d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/kernel*&
_output_shapes
:@0*
dtype0
?
conv2d_transpose_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_8/bias

+conv2d_transpose_8/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_transpose_9/kernel
?
-conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/kernel*&
_output_shapes
:@*
dtype0
?
conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_9/bias

+conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
?
conv2d_4/p_re_lu_8/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?? *)
shared_nameconv2d_4/p_re_lu_8/alpha
?
,conv2d_4/p_re_lu_8/alpha/Read/ReadVariableOpReadVariableOpconv2d_4/p_re_lu_8/alpha*$
_output_shapes
:?? *
dtype0
?
conv2d_5/p_re_lu_9/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:??0*)
shared_nameconv2d_5/p_re_lu_9/alpha
?
,conv2d_5/p_re_lu_9/alpha/Read/ReadVariableOpReadVariableOpconv2d_5/p_re_lu_9/alpha*$
_output_shapes
:??0*
dtype0
?
conv2d_6/p_re_lu_10/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@@**
shared_nameconv2d_6/p_re_lu_10/alpha
?
-conv2d_6/p_re_lu_10/alpha/Read/ReadVariableOpReadVariableOpconv2d_6/p_re_lu_10/alpha*"
_output_shapes
:@@@*
dtype0
?
conv2d_7/p_re_lu_11/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameconv2d_7/p_re_lu_11/alpha
?
-conv2d_7/p_re_lu_11/alpha/Read/ReadVariableOpReadVariableOpconv2d_7/p_re_lu_11/alpha*"
_output_shapes
:  *
dtype0
?
#conv2d_transpose_5/p_re_lu_12/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#conv2d_transpose_5/p_re_lu_12/alpha
?
7conv2d_transpose_5/p_re_lu_12/alpha/Read/ReadVariableOpReadVariableOp#conv2d_transpose_5/p_re_lu_12/alpha*"
_output_shapes
:@@*
dtype0
?
#conv2d_transpose_6/p_re_lu_13/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?? *4
shared_name%#conv2d_transpose_6/p_re_lu_13/alpha
?
7conv2d_transpose_6/p_re_lu_13/alpha/Read/ReadVariableOpReadVariableOp#conv2d_transpose_6/p_re_lu_13/alpha*$
_output_shapes
:?? *
dtype0
?
#conv2d_transpose_7/p_re_lu_14/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:??0*4
shared_name%#conv2d_transpose_7/p_re_lu_14/alpha
?
7conv2d_transpose_7/p_re_lu_14/alpha/Read/ReadVariableOpReadVariableOp#conv2d_transpose_7/p_re_lu_14/alpha*$
_output_shapes
:??0*
dtype0
?
#conv2d_transpose_8/p_re_lu_15/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:??@*4
shared_name%#conv2d_transpose_8/p_re_lu_15/alpha
?
7conv2d_transpose_8/p_re_lu_15/alpha/Read/ReadVariableOpReadVariableOp#conv2d_transpose_8/p_re_lu_15/alpha*$
_output_shapes
:??@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
layer_with_weights-16
layer-17
	optimizer
loss
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
x

activation

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
?
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&trainable_variables
'	variables
(regularization_losses
)	keras_api
x
*
activation

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6trainable_variables
7	variables
8regularization_losses
9	keras_api
x
:
activation

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
x
J
activation

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
x
Z
activation

[kernel
\bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
?
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
x
j
activation

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
y
z
activation

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?
activation
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
&
	?iter

?decay
?momentum
 
?
0
1
?2
"3
#4
+5
,6
?7
28
39
;10
<11
?12
B13
C14
K15
L16
?17
R18
S19
[20
\21
?22
b23
c24
k25
l26
?27
r28
s29
{30
|31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?
0
1
?2
"3
#4
$5
%6
+7
,8
?9
210
311
412
513
;14
<15
?16
B17
C18
D19
E20
K21
L22
?23
R24
S25
T26
U27
[28
\29
?30
b31
c32
d33
e34
k35
l36
?37
r38
s39
t40
u41
{42
|43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
 
?
?layer_metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?metrics
?layers
 
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
?2

0
1
?2
 
?
?layer_metrics
trainable_variables
 ?layer_regularization_losses
	variables
?non_trainable_variables
regularization_losses
?metrics
?layers
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
$2
%3
 
?
?layer_metrics
&trainable_variables
 ?layer_regularization_losses
'	variables
?non_trainable_variables
(regularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
?2

+0
,1
?2
 
?
?layer_metrics
-trainable_variables
 ?layer_regularization_losses
.	variables
?non_trainable_variables
/regularization_losses
?metrics
?layers
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
42
53
 
?
?layer_metrics
6trainable_variables
 ?layer_regularization_losses
7	variables
?non_trainable_variables
8regularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
?2

;0
<1
?2
 
?
?layer_metrics
=trainable_variables
 ?layer_regularization_losses
>	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
D2
E3
 
?
?layer_metrics
Ftrainable_variables
 ?layer_regularization_losses
G	variables
?non_trainable_variables
Hregularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
?2

K0
L1
?2
 
?
?layer_metrics
Mtrainable_variables
 ?layer_regularization_losses
N	variables
?non_trainable_variables
Oregularization_losses
?metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
T2
U3
 
?
?layer_metrics
Vtrainable_variables
 ?layer_regularization_losses
W	variables
?non_trainable_variables
Xregularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
ec
VARIABLE_VALUEconv2d_transpose_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
?2

[0
\1
?2
 
?
?layer_metrics
]trainable_variables
 ?layer_regularization_losses
^	variables
?non_trainable_variables
_regularization_losses
?metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

b0
c1
d2
e3
 
?
?layer_metrics
ftrainable_variables
 ?layer_regularization_losses
g	variables
?non_trainable_variables
hregularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
fd
VARIABLE_VALUEconv2d_transpose_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
?2

k0
l1
?2
 
?
?layer_metrics
mtrainable_variables
 ?layer_regularization_losses
n	variables
?non_trainable_variables
oregularization_losses
?metrics
?layers
 
hf
VARIABLE_VALUEbatch_normalization_13/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_13/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_13/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_13/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

r0
s1
t2
u3
 
?
?layer_metrics
vtrainable_variables
 ?layer_regularization_losses
w	variables
?non_trainable_variables
xregularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
fd
VARIABLE_VALUEconv2d_transpose_7/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_7/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
?2

{0
|1
?2
 
?
?layer_metrics
}trainable_variables
 ?layer_regularization_losses
~	variables
?non_trainable_variables
regularization_losses
?metrics
?layers
 
hf
VARIABLE_VALUEbatch_normalization_14/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_14/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_14/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_14/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
?0
?1
?2
?3
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
b

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
fd
VARIABLE_VALUEconv2d_transpose_8/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_8/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
?2

?0
?1
?2
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
hf
VARIABLE_VALUEbatch_normalization_15/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_15/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_15/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_15/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
?0
?1
?2
?3
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
fd
VARIABLE_VALUEconv2d_transpose_9/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_9/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_4/p_re_lu_8/alpha0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_5/p_re_lu_9/alpha0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_6/p_re_lu_10/alpha1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_7/p_re_lu_11/alpha1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#conv2d_transpose_5/p_re_lu_12/alpha1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#conv2d_transpose_6/p_re_lu_13/alpha1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#conv2d_transpose_7/p_re_lu_14/alpha1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#conv2d_transpose_8/p_re_lu_15/alpha1trainable_variables/37/.ATTRIBUTES/VARIABLE_VALUE
 
 
z
$0
%1
42
53
D4
E5
T6
U7
d8
e9
t10
u11
?12
?13
?14
?15
 
?0
?1
?2
?3
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

0
 
 

$0
%1
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

*0
 
 

40
51
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

:0
 
 

D0
E1
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

J0
 
 

T0
U1
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

Z0
 
 

d0
e1
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

j0
 
 

t0
u1
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

z0
 
 

?0
?1
 
 

?0

?0
 
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
 
 
 
 

?0
 
 

?0
?1
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_4/kernelconv2d_4/biasconv2d_4/p_re_lu_8/alphabatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_5/kernelconv2d_5/biasconv2d_5/p_re_lu_9/alphabatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_6/p_re_lu_10/alphabatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_7/kernelconv2d_7/biasconv2d_7/p_re_lu_11/alphabatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/bias#conv2d_transpose_5/p_re_lu_12/alphabatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_transpose_6/kernelconv2d_transpose_6/bias#conv2d_transpose_6/p_re_lu_13/alphabatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_transpose_7/kernelconv2d_transpose_7/bias#conv2d_transpose_7/p_re_lu_14/alphabatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_transpose_8/kernelconv2d_transpose_8/bias#conv2d_transpose_8/p_re_lu_15/alphabatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_transpose_9/kernelconv2d_transpose_9/bias*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2421055
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp-conv2d_transpose_6/kernel/Read/ReadVariableOp+conv2d_transpose_6/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp-conv2d_transpose_7/kernel/Read/ReadVariableOp+conv2d_transpose_7/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp-conv2d_transpose_8/kernel/Read/ReadVariableOp+conv2d_transpose_8/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp-conv2d_transpose_9/kernel/Read/ReadVariableOp+conv2d_transpose_9/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp,conv2d_4/p_re_lu_8/alpha/Read/ReadVariableOp,conv2d_5/p_re_lu_9/alpha/Read/ReadVariableOp-conv2d_6/p_re_lu_10/alpha/Read/ReadVariableOp-conv2d_7/p_re_lu_11/alpha/Read/ReadVariableOp7conv2d_transpose_5/p_re_lu_12/alpha/Read/ReadVariableOp7conv2d_transpose_6/p_re_lu_13/alpha/Read/ReadVariableOp7conv2d_transpose_7/p_re_lu_14/alpha/Read/ReadVariableOp7conv2d_transpose_8/p_re_lu_15/alpha/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2423915
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_transpose_6/kernelconv2d_transpose_6/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_transpose_7/kernelconv2d_transpose_7/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_transpose_8/kernelconv2d_transpose_8/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_transpose_9/kernelconv2d_transpose_9/biasSGD/iter	SGD/decaySGD/momentumconv2d_4/p_re_lu_8/alphaconv2d_5/p_re_lu_9/alphaconv2d_6/p_re_lu_10/alphaconv2d_7/p_re_lu_11/alpha#conv2d_transpose_5/p_re_lu_12/alpha#conv2d_transpose_6/p_re_lu_13/alpha#conv2d_transpose_7/p_re_lu_14/alpha#conv2d_transpose_8/p_re_lu_15/alphatotalcounttotal_1count_1total_2count_2total_3count_3*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2424132γ*
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423293

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?&
?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2419025

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421998

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_14_layer_call_fn_2423124

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_24186022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_2419703
input_2!
unknown: 
	unknown_0: !
	unknown_1:?? 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: 0
	unknown_7:0!
	unknown_8:??0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@ 

unknown_15:@@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@

unknown_21: 

unknown_22:  

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:

unknown_28: 

unknown_29:@@

unknown_30:

unknown_31:

unknown_32:

unknown_33:$

unknown_34: 

unknown_35: "

unknown_36:?? 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:0 

unknown_42:0"

unknown_43:??0

unknown_44:0

unknown_45:0

unknown_46:0

unknown_47:0$

unknown_48:@0

unknown_49:@"

unknown_50:??@

unknown_51:@

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_24195842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?)
?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2418798

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@*
p_re_lu_15_2418794:??@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?"p_re_lu_15/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
"p_re_lu_15/StatefulPartitionedCallStatefulPartitionedCallBiasAdd:output:0p_re_lu_15_2418794*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_24187932$
"p_re_lu_15/StatefulPartitionedCall?
IdentityIdentity+p_re_lu_15/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp#^p_re_lu_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????0: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2H
"p_re_lu_15/StatefulPartitionedCall"p_re_lu_15/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2418883

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?z
?
D__inference_model_1_layer_call_and_return_conditional_losses_2420930
input_2*
conv2d_4_2420796: 
conv2d_4_2420798: (
conv2d_4_2420800:?? +
batch_normalization_8_2420803: +
batch_normalization_8_2420805: +
batch_normalization_8_2420807: +
batch_normalization_8_2420809: *
conv2d_5_2420812: 0
conv2d_5_2420814:0(
conv2d_5_2420816:??0+
batch_normalization_9_2420819:0+
batch_normalization_9_2420821:0+
batch_normalization_9_2420823:0+
batch_normalization_9_2420825:0*
conv2d_6_2420828:0@
conv2d_6_2420830:@&
conv2d_6_2420832:@@@,
batch_normalization_10_2420835:@,
batch_normalization_10_2420837:@,
batch_normalization_10_2420839:@,
batch_normalization_10_2420841:@*
conv2d_7_2420844:@
conv2d_7_2420846:&
conv2d_7_2420848:  ,
batch_normalization_11_2420851:,
batch_normalization_11_2420853:,
batch_normalization_11_2420855:,
batch_normalization_11_2420857:4
conv2d_transpose_5_2420860:(
conv2d_transpose_5_2420862:0
conv2d_transpose_5_2420864:@@,
batch_normalization_12_2420867:,
batch_normalization_12_2420869:,
batch_normalization_12_2420871:,
batch_normalization_12_2420873:4
conv2d_transpose_6_2420876: (
conv2d_transpose_6_2420878: 2
conv2d_transpose_6_2420880:?? ,
batch_normalization_13_2420883: ,
batch_normalization_13_2420885: ,
batch_normalization_13_2420887: ,
batch_normalization_13_2420889: 4
conv2d_transpose_7_2420892:0 (
conv2d_transpose_7_2420894:02
conv2d_transpose_7_2420896:??0,
batch_normalization_14_2420899:0,
batch_normalization_14_2420901:0,
batch_normalization_14_2420903:0,
batch_normalization_14_2420905:04
conv2d_transpose_8_2420908:@0(
conv2d_transpose_8_2420910:@2
conv2d_transpose_8_2420912:??@,
batch_normalization_15_2420915:@,
batch_normalization_15_2420917:@,
batch_normalization_15_2420919:@,
batch_normalization_15_2420921:@4
conv2d_transpose_9_2420924:@(
conv2d_transpose_9_2420926:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_4_2420796conv2d_4_2420798conv2d_4_2420800*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_24191002"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_8_2420803batch_normalization_8_2420805batch_normalization_8_2420807batch_normalization_8_2420809*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24201332/
-batch_normalization_8/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_5_2420812conv2d_5_2420814conv2d_5_2420816*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_24191532"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_2420819batch_normalization_9_2420821batch_normalization_9_2420823batch_normalization_9_2420825*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24200772/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_6_2420828conv2d_6_2420830conv2d_6_2420832*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_24192062"
 conv2d_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_2420835batch_normalization_10_2420837batch_normalization_10_2420839batch_normalization_10_2420841*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_242002120
.batch_normalization_10/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_7_2420844conv2d_7_2420846conv2d_7_2420848*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_24192592"
 conv2d_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_2420851batch_normalization_11_2420853batch_normalization_11_2420855batch_normalization_11_2420857*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_241996520
.batch_normalization_11/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_5_2420860conv2d_transpose_5_2420862conv2d_transpose_5_2420864*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_24193242,
*conv2d_transpose_5/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_12_2420867batch_normalization_12_2420869batch_normalization_12_2420871batch_normalization_12_2420873*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_241991020
.batch_normalization_12/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_transpose_6_2420876conv2d_transpose_6_2420878conv2d_transpose_6_2420880*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_24193892,
*conv2d_transpose_6/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_13_2420883batch_normalization_13_2420885batch_normalization_13_2420887batch_normalization_13_2420889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_241985520
.batch_normalization_13/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_transpose_7_2420892conv2d_transpose_7_2420894conv2d_transpose_7_2420896*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_24194542,
*conv2d_transpose_7/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_14_2420899batch_normalization_14_2420901batch_normalization_14_2420903batch_normalization_14_2420905*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_241980020
.batch_normalization_14/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_transpose_8_2420908conv2d_transpose_8_2420910conv2d_transpose_8_2420912*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_24195192,
*conv2d_transpose_8/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_15_2420915batch_normalization_15_2420917batch_normalization_15_2420919batch_normalization_15_2420921*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_241974520
.batch_normalization_15/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv2d_transpose_9_2420924conv2d_transpose_9_2420926*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_24195772,
*conv2d_transpose_9/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2423583

inputs/
readvariableop_resource:?? 
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:?? *
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:??????????? 2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2419284

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2423488

inputs/
readvariableop_resource:??0
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??0*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??02
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????02
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_2422024

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24173052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_12_layer_call_fn_2422701

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_24180842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2417381

inputs/
readvariableop_resource:??0
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??0*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??02
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????02
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_2421776

inputs!
unknown: 
	unknown_0: !
	unknown_1:?? 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: 0
	unknown_7:0!
	unknown_8:??0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@ 

unknown_15:@@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@

unknown_21: 

unknown_22:  

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:

unknown_28: 

unknown_29:@@

unknown_30:

unknown_31:

unknown_32:

unknown_33:$

unknown_34: 

unknown_35: "

unknown_36:?? 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:0 

unknown_42:0"

unknown_43:??0

unknown_44:0

unknown_45:0

unknown_46:0

unknown_47:0$

unknown_48:@0

unknown_49:@"

unknown_50:??@

unknown_51:@

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_24195842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2419965

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_15_layer_call_fn_2423355

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_24189272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2417547

inputs-
readvariableop_resource:@@@
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:@@@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????@@@2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2417305

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_13_layer_call_fn_2423609

inputs
unknown:?? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_24182312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+??????????????????????????? : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_7_layer_call_fn_2422385

inputs!
unknown:@
	unknown_0:
	unknown_1:  
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_24192592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@@@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2423671

inputs/
readvariableop_resource:??@
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??@*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????@2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_conv2d_4_layer_call_fn_2421926

inputs!
unknown: 
	unknown_0: !
	unknown_1:?? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_24191002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2419544

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_11_layer_call_fn_2422483

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_24178032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_p_re_lu_9_layer_call_fn_2423495

inputs
unknown:??0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_24173812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2419800

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2422374

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:8
"p_re_lu_11_readvariableop_resource:  
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_11/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAddv
p_re_lu_11/ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/Relu?
p_re_lu_11/ReadVariableOpReadVariableOp"p_re_lu_11_readvariableop_resource*"
_output_shapes
:  *
dtype02
p_re_lu_11/ReadVariableOpw
p_re_lu_11/NegNeg!p_re_lu_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2
p_re_lu_11/Negw
p_re_lu_11/Neg_1NegBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/Neg_1~
p_re_lu_11/Relu_1Relup_re_lu_11/Neg_1:y:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/Relu_1?
p_re_lu_11/mulMulp_re_lu_11/Neg:y:0p_re_lu_11/Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/mul?
p_re_lu_11/addAddV2p_re_lu_11/Relu:activations:0p_re_lu_11/mul:z:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/addu
IdentityIdentityp_re_lu_11/add:z:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@@@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp26
p_re_lu_11/ReadVariableOpp_re_lu_11/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?"
?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2422799

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: :
"p_re_lu_13_readvariableop_resource:?? 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_13/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddx
p_re_lu_13/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/Relu?
p_re_lu_13/ReadVariableOpReadVariableOp"p_re_lu_13_readvariableop_resource*$
_output_shapes
:?? *
dtype02
p_re_lu_13/ReadVariableOpy
p_re_lu_13/NegNeg!p_re_lu_13/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
p_re_lu_13/Negy
p_re_lu_13/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/Neg_1?
p_re_lu_13/Relu_1Relup_re_lu_13/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/Relu_1?
p_re_lu_13/mulMulp_re_lu_13/Neg:y:0p_re_lu_13/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/mul?
p_re_lu_13/addAddV2p_re_lu_13/Relu:activations:0p_re_lu_13/mul:z:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/addw
IdentityIdentityp_re_lu_13/add:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_13/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_13/ReadVariableOpp_re_lu_13/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_13_layer_call_fn_2422906

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_24183212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?)
?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2417955

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:(
p_re_lu_12_2417951:@@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?"p_re_lu_12/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
"p_re_lu_12/StatefulPartitionedCallStatefulPartitionedCallBiasAdd:output:0p_re_lu_12_2417951*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_24179502$
"p_re_lu_12/StatefulPartitionedCall?
IdentityIdentity+p_re_lu_12/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp#^p_re_lu_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2H
"p_re_lu_12/StatefulPartitionedCall"p_re_lu_12/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2418160

inputs/
readvariableop_resource:?? 
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:?? *
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:??????????? 2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_8_layer_call_fn_2423257

inputs!
unknown:@0
	unknown_0:@!
	unknown_1:??@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_24195192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????0: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2419206

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@8
"p_re_lu_10_readvariableop_resource:@@@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_10/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAddv
p_re_lu_10/ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/Relu?
p_re_lu_10/ReadVariableOpReadVariableOp"p_re_lu_10_readvariableop_resource*"
_output_shapes
:@@@*
dtype02
p_re_lu_10/ReadVariableOpw
p_re_lu_10/NegNeg!p_re_lu_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2
p_re_lu_10/Negw
p_re_lu_10/Neg_1NegBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/Neg_1~
p_re_lu_10/Relu_1Relup_re_lu_10/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/Relu_1?
p_re_lu_10/mulMulp_re_lu_10/Neg:y:0p_re_lu_10/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/mul?
p_re_lu_10/addAddV2p_re_lu_10/Relu:activations:0p_re_lu_10/mul:z:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/addu
IdentityIdentityp_re_lu_10/add:z:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????0: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp26
p_re_lu_10/ReadVariableOpp_re_lu_10/ReadVariableOp:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
+__inference_p_re_lu_8_layer_call_fn_2423476

inputs
unknown:?? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_24172152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_11_layer_call_fn_2423533

inputs
unknown:  
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_24177132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2419125

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2421055
input_2!
unknown: 
	unknown_0: !
	unknown_1:?? 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: 0
	unknown_7:0!
	unknown_8:??0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@ 

unknown_15:@@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@

unknown_21: 

unknown_22:  

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:

unknown_28: 

unknown_29:@@

unknown_30:

unknown_31:

unknown_32:

unknown_33:$

unknown_34: 

unknown_35: "

unknown_36:?? 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:0 

unknown_42:0"

unknown_43:??0

unknown_44:0

unknown_45:0

unknown_46:0

unknown_47:0$

unknown_48:@0

unknown_49:@"

unknown_50:??@

unknown_51:@

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_24171992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
,__inference_p_re_lu_12_layer_call_fn_2423564

inputs
unknown:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_24178792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_14_layer_call_fn_2423647

inputs
unknown:??0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_24185122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????0: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421962

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423111

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?.
?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2422986

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0:
"p_re_lu_14_readvariableop_resource:??0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_14/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :02	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????02	
BiasAdd?
p_re_lu_14/ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02
p_re_lu_14/Relu?
p_re_lu_14/ReadVariableOpReadVariableOp"p_re_lu_14_readvariableop_resource*$
_output_shapes
:??0*
dtype02
p_re_lu_14/ReadVariableOpy
p_re_lu_14/NegNeg!p_re_lu_14/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
p_re_lu_14/Neg?
p_re_lu_14/Neg_1NegBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02
p_re_lu_14/Neg_1?
p_re_lu_14/Relu_1Relup_re_lu_14/Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02
p_re_lu_14/Relu_1?
p_re_lu_14/mulMulp_re_lu_14/Neg:y:0p_re_lu_14/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/mul?
p_re_lu_14/addAddV2p_re_lu_14/Relu:activations:0p_re_lu_14/mul:z:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/addw
IdentityIdentityp_re_lu_14/add:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+??????????????????????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_14/ReadVariableOpp_re_lu_14/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422857

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422875

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423311

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422115

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2418040

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_5_layer_call_fn_2422079

inputs!
unknown: 0
	unknown_0:0!
	unknown_1:??0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_24191532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_2421897

inputs!
unknown: 
	unknown_0: !
	unknown_1:?? 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: 0
	unknown_7:0!
	unknown_8:??0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@ 

unknown_15:@@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@

unknown_21: 

unknown_22:  

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:

unknown_28: 

unknown_29:@@

unknown_30:

unknown_31:

unknown_32:

unknown_33:$

unknown_34: 

unknown_35: "

unknown_36:?? 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:0 

unknown_42:0"

unknown_43:??0

unknown_44:0

unknown_45:0

unknown_46:0

unknown_47:0$

unknown_48:@0

unknown_49:@"

unknown_50:??@

unknown_51:@

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*L
_read_only_resource_inputs.
,*	
 !$%&'(+,-./234569:*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_24204162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2417261

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_13_layer_call_fn_2422945

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_24198552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422457

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2418602

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2419153

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:09
!p_re_lu_9_readvariableop_resource:??0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_9/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddv
p_re_lu_9/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/Relu?
p_re_lu_9/ReadVariableOpReadVariableOp!p_re_lu_9_readvariableop_resource*$
_output_shapes
:??0*
dtype02
p_re_lu_9/ReadVariableOpv
p_re_lu_9/NegNeg p_re_lu_9/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
p_re_lu_9/Negw
p_re_lu_9/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/Neg_1}
p_re_lu_9/Relu_1Relup_re_lu_9/Neg_1:y:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/Relu_1?
p_re_lu_9/mulMulp_re_lu_9/Neg:y:0p_re_lu_9/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/mul?
p_re_lu_9/addAddV2p_re_lu_9/Relu:activations:0p_re_lu_9/mul:z:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/addv
IdentityIdentityp_re_lu_9/add:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp24
p_re_lu_9/ReadVariableOpp_re_lu_9/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_2422037

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24191252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?"
?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2423017

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0:
"p_re_lu_14_readvariableop_resource:??0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_14/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :02	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddx
p_re_lu_14/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/Relu?
p_re_lu_14/ReadVariableOpReadVariableOp"p_re_lu_14_readvariableop_resource*$
_output_shapes
:??0*
dtype02
p_re_lu_14/ReadVariableOpy
p_re_lu_14/NegNeg!p_re_lu_14/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
p_re_lu_14/Negy
p_re_lu_14/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/Neg_1?
p_re_lu_14/Relu_1Relup_re_lu_14/Neg_1:y:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/Relu_1?
p_re_lu_14/mulMulp_re_lu_14/Neg:y:0p_re_lu_14/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/mul?
p_re_lu_14/addAddV2p_re_lu_14/Relu:activations:0p_re_lu_14/mul:z:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/addw
IdentityIdentityp_re_lu_14/add:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_14/ReadVariableOpp_re_lu_14/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_15_layer_call_fn_2423342

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_24188832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_12_layer_call_fn_2422688

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_24180402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_10_layer_call_fn_2422317

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_24175932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2418722

inputs/
readvariableop_resource:??@
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??@*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????@2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_10_layer_call_fn_2422356

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_24200212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2418646

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423275

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
4__inference_conv2d_transpose_8_layer_call_fn_2423246

inputs!
unknown:@0
	unknown_0:@!
	unknown_1:??@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_24187982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????0: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2423439

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoidp
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422268

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2423507

inputs-
readvariableop_resource:@@@
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:@@@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????@@@2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2418512

inputs/
readvariableop_resource:??0
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????02
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??0*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??02
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+???????????????????????????02
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????02
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????0: 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_13_layer_call_fn_2422932

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_24194142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422893

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_14_layer_call_fn_2423150

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_24194792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?"
?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2419389

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: :
"p_re_lu_13_readvariableop_resource:?? 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_13/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddx
p_re_lu_13/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/Relu?
p_re_lu_13/ReadVariableOpReadVariableOp"p_re_lu_13_readvariableop_resource*$
_output_shapes
:?? *
dtype02
p_re_lu_13/ReadVariableOpy
p_re_lu_13/NegNeg!p_re_lu_13/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
p_re_lu_13/Negy
p_re_lu_13/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/Neg_1?
p_re_lu_13/Relu_1Relup_re_lu_13/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/Relu_1?
p_re_lu_13/mulMulp_re_lu_13/Neg:y:0p_re_lu_13/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/mul?
p_re_lu_13/addAddV2p_re_lu_13/Relu:activations:0p_re_lu_13/mul:z:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/addw
IdentityIdentityp_re_lu_13/add:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_13/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_13/ReadVariableOpp_re_lu_13/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?"
?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2419324

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:8
"p_re_lu_12_readvariableop_resource:@@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_12/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAddv
p_re_lu_12/ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/Relu?
p_re_lu_12/ReadVariableOpReadVariableOp"p_re_lu_12_readvariableop_resource*"
_output_shapes
:@@*
dtype02
p_re_lu_12/ReadVariableOpw
p_re_lu_12/NegNeg!p_re_lu_12/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
p_re_lu_12/Negw
p_re_lu_12/Neg_1NegBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/Neg_1~
p_re_lu_12/Relu_1Relup_re_lu_12/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/Relu_1?
p_re_lu_12/mulMulp_re_lu_12/Neg:y:0p_re_lu_12/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/mul?
p_re_lu_12/addAddV2p_re_lu_12/Relu:activations:0p_re_lu_12/mul:z:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/addu
IdentityIdentityp_re_lu_12/add:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_12/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????  : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_12/ReadVariableOpp_re_lu_12/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?)
?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2418236

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: *
p_re_lu_13_2418232:?? 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?"p_re_lu_13/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
"p_re_lu_13/StatefulPartitionedCallStatefulPartitionedCallBiasAdd:output:0p_re_lu_13_2418232*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_24182312$
"p_re_lu_13/StatefulPartitionedCall?
IdentityIdentity+p_re_lu_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp#^p_re_lu_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2H
"p_re_lu_13/StatefulPartitionedCall"p_re_lu_13/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?.
?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2423204

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@:
"p_re_lu_15_readvariableop_resource:??@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_15/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
p_re_lu_15/ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
p_re_lu_15/Relu?
p_re_lu_15/ReadVariableOpReadVariableOp"p_re_lu_15_readvariableop_resource*$
_output_shapes
:??@*
dtype02
p_re_lu_15/ReadVariableOpy
p_re_lu_15/NegNeg!p_re_lu_15/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
p_re_lu_15/Neg?
p_re_lu_15/Neg_1NegBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
p_re_lu_15/Neg_1?
p_re_lu_15/Relu_1Relup_re_lu_15/Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
p_re_lu_15/Relu_1?
p_re_lu_15/mulMulp_re_lu_15/Neg:y:0p_re_lu_15/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/mul?
p_re_lu_15/addAddV2p_re_lu_15/Relu:activations:0p_re_lu_15/mul:z:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/addw
IdentityIdentityp_re_lu_15/add:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????0: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_15/ReadVariableOpp_re_lu_15/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_14_layer_call_fn_2423640

inputs
unknown:??0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_24184412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_2419100

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 9
!p_re_lu_8_readvariableop_resource:?? 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_8/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddv
p_re_lu_8/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/Relu?
p_re_lu_8/ReadVariableOpReadVariableOp!p_re_lu_8_readvariableop_resource*$
_output_shapes
:?? *
dtype02
p_re_lu_8/ReadVariableOpv
p_re_lu_8/NegNeg p_re_lu_8/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
p_re_lu_8/Negw
p_re_lu_8/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/Neg_1}
p_re_lu_8/Relu_1Relup_re_lu_8/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/Relu_1?
p_re_lu_8/mulMulp_re_lu_8/Neg:y:0p_re_lu_8/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/mul?
p_re_lu_8/addAddV2p_re_lu_8/Relu:activations:0p_re_lu_8/mul:z:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/addv
IdentityIdentityp_re_lu_8/add:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp24
p_re_lu_8/ReadVariableOpp_re_lu_8/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2423595

inputs/
readvariableop_resource:?? 
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:?? *
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:??????????? 2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+??????????????????????????? : 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?z
?
D__inference_model_1_layer_call_and_return_conditional_losses_2419584

inputs*
conv2d_4_2419101: 
conv2d_4_2419103: (
conv2d_4_2419105:?? +
batch_normalization_8_2419126: +
batch_normalization_8_2419128: +
batch_normalization_8_2419130: +
batch_normalization_8_2419132: *
conv2d_5_2419154: 0
conv2d_5_2419156:0(
conv2d_5_2419158:??0+
batch_normalization_9_2419179:0+
batch_normalization_9_2419181:0+
batch_normalization_9_2419183:0+
batch_normalization_9_2419185:0*
conv2d_6_2419207:0@
conv2d_6_2419209:@&
conv2d_6_2419211:@@@,
batch_normalization_10_2419232:@,
batch_normalization_10_2419234:@,
batch_normalization_10_2419236:@,
batch_normalization_10_2419238:@*
conv2d_7_2419260:@
conv2d_7_2419262:&
conv2d_7_2419264:  ,
batch_normalization_11_2419285:,
batch_normalization_11_2419287:,
batch_normalization_11_2419289:,
batch_normalization_11_2419291:4
conv2d_transpose_5_2419325:(
conv2d_transpose_5_2419327:0
conv2d_transpose_5_2419329:@@,
batch_normalization_12_2419350:,
batch_normalization_12_2419352:,
batch_normalization_12_2419354:,
batch_normalization_12_2419356:4
conv2d_transpose_6_2419390: (
conv2d_transpose_6_2419392: 2
conv2d_transpose_6_2419394:?? ,
batch_normalization_13_2419415: ,
batch_normalization_13_2419417: ,
batch_normalization_13_2419419: ,
batch_normalization_13_2419421: 4
conv2d_transpose_7_2419455:0 (
conv2d_transpose_7_2419457:02
conv2d_transpose_7_2419459:??0,
batch_normalization_14_2419480:0,
batch_normalization_14_2419482:0,
batch_normalization_14_2419484:0,
batch_normalization_14_2419486:04
conv2d_transpose_8_2419520:@0(
conv2d_transpose_8_2419522:@2
conv2d_transpose_8_2419524:??@,
batch_normalization_15_2419545:@,
batch_normalization_15_2419547:@,
batch_normalization_15_2419549:@,
batch_normalization_15_2419551:@4
conv2d_transpose_9_2419578:@(
conv2d_transpose_9_2419580:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_2419101conv2d_4_2419103conv2d_4_2419105*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_24191002"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_8_2419126batch_normalization_8_2419128batch_normalization_8_2419130batch_normalization_8_2419132*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24191252/
-batch_normalization_8/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_5_2419154conv2d_5_2419156conv2d_5_2419158*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_24191532"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_2419179batch_normalization_9_2419181batch_normalization_9_2419183batch_normalization_9_2419185*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24191782/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_6_2419207conv2d_6_2419209conv2d_6_2419211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_24192062"
 conv2d_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_2419232batch_normalization_10_2419234batch_normalization_10_2419236batch_normalization_10_2419238*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_241923120
.batch_normalization_10/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_7_2419260conv2d_7_2419262conv2d_7_2419264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_24192592"
 conv2d_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_2419285batch_normalization_11_2419287batch_normalization_11_2419289batch_normalization_11_2419291*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_241928420
.batch_normalization_11/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_5_2419325conv2d_transpose_5_2419327conv2d_transpose_5_2419329*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_24193242,
*conv2d_transpose_5/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_12_2419350batch_normalization_12_2419352batch_normalization_12_2419354batch_normalization_12_2419356*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_241934920
.batch_normalization_12/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_transpose_6_2419390conv2d_transpose_6_2419392conv2d_transpose_6_2419394*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_24193892,
*conv2d_transpose_6/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_13_2419415batch_normalization_13_2419417batch_normalization_13_2419419batch_normalization_13_2419421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_241941420
.batch_normalization_13/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_transpose_7_2419455conv2d_transpose_7_2419457conv2d_transpose_7_2419459*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_24194542,
*conv2d_transpose_7/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_14_2419480batch_normalization_14_2419482batch_normalization_14_2419484batch_normalization_14_2419486*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_241947920
.batch_normalization_14/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_transpose_8_2419520conv2d_transpose_8_2419522conv2d_transpose_8_2419524*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_24195192,
*conv2d_transpose_8/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_15_2419545batch_normalization_15_2419547batch_normalization_15_2419549batch_normalization_15_2419551*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_241954420
.batch_normalization_15/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv2d_transpose_9_2419578conv2d_transpose_9_2419580*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_24195772,
*conv2d_transpose_9/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?"
?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2422581

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:8
"p_re_lu_12_readvariableop_resource:@@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_12/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAddv
p_re_lu_12/ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/Relu?
p_re_lu_12/ReadVariableOpReadVariableOp"p_re_lu_12_readvariableop_resource*"
_output_shapes
:@@*
dtype02
p_re_lu_12/ReadVariableOpw
p_re_lu_12/NegNeg!p_re_lu_12/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
p_re_lu_12/Negw
p_re_lu_12/Neg_1NegBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/Neg_1~
p_re_lu_12/Relu_1Relup_re_lu_12/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/Relu_1?
p_re_lu_12/mulMulp_re_lu_12/Neg:y:0p_re_lu_12/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/mul?
p_re_lu_12/addAddV2p_re_lu_12/Relu:activations:0p_re_lu_12/mul:z:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/addu
IdentityIdentityp_re_lu_12/add:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_12/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????  : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_12/ReadVariableOpp_re_lu_12/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
*__inference_conv2d_6_layer_call_fn_2422232

inputs!
unknown:0@
	unknown_0:@
	unknown_1:@@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_24192062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????0: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423093

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2417637

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2423633

inputs/
readvariableop_resource:??0
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????02
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??0*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??02
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+???????????????????????????02
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????02
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????0: 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422421

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?z
?
D__inference_model_1_layer_call_and_return_conditional_losses_2420793
input_2*
conv2d_4_2420659: 
conv2d_4_2420661: (
conv2d_4_2420663:?? +
batch_normalization_8_2420666: +
batch_normalization_8_2420668: +
batch_normalization_8_2420670: +
batch_normalization_8_2420672: *
conv2d_5_2420675: 0
conv2d_5_2420677:0(
conv2d_5_2420679:??0+
batch_normalization_9_2420682:0+
batch_normalization_9_2420684:0+
batch_normalization_9_2420686:0+
batch_normalization_9_2420688:0*
conv2d_6_2420691:0@
conv2d_6_2420693:@&
conv2d_6_2420695:@@@,
batch_normalization_10_2420698:@,
batch_normalization_10_2420700:@,
batch_normalization_10_2420702:@,
batch_normalization_10_2420704:@*
conv2d_7_2420707:@
conv2d_7_2420709:&
conv2d_7_2420711:  ,
batch_normalization_11_2420714:,
batch_normalization_11_2420716:,
batch_normalization_11_2420718:,
batch_normalization_11_2420720:4
conv2d_transpose_5_2420723:(
conv2d_transpose_5_2420725:0
conv2d_transpose_5_2420727:@@,
batch_normalization_12_2420730:,
batch_normalization_12_2420732:,
batch_normalization_12_2420734:,
batch_normalization_12_2420736:4
conv2d_transpose_6_2420739: (
conv2d_transpose_6_2420741: 2
conv2d_transpose_6_2420743:?? ,
batch_normalization_13_2420746: ,
batch_normalization_13_2420748: ,
batch_normalization_13_2420750: ,
batch_normalization_13_2420752: 4
conv2d_transpose_7_2420755:0 (
conv2d_transpose_7_2420757:02
conv2d_transpose_7_2420759:??0,
batch_normalization_14_2420762:0,
batch_normalization_14_2420764:0,
batch_normalization_14_2420766:0,
batch_normalization_14_2420768:04
conv2d_transpose_8_2420771:@0(
conv2d_transpose_8_2420773:@2
conv2d_transpose_8_2420775:??@,
batch_normalization_15_2420778:@,
batch_normalization_15_2420780:@,
batch_normalization_15_2420782:@,
batch_normalization_15_2420784:@4
conv2d_transpose_9_2420787:@(
conv2d_transpose_9_2420789:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_4_2420659conv2d_4_2420661conv2d_4_2420663*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_24191002"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_8_2420666batch_normalization_8_2420668batch_normalization_8_2420670batch_normalization_8_2420672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24191252/
-batch_normalization_8/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_5_2420675conv2d_5_2420677conv2d_5_2420679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_24191532"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_2420682batch_normalization_9_2420684batch_normalization_9_2420686batch_normalization_9_2420688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24191782/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_6_2420691conv2d_6_2420693conv2d_6_2420695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_24192062"
 conv2d_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_2420698batch_normalization_10_2420700batch_normalization_10_2420702batch_normalization_10_2420704*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_241923120
.batch_normalization_10/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_7_2420707conv2d_7_2420709conv2d_7_2420711*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_24192592"
 conv2d_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_2420714batch_normalization_11_2420716batch_normalization_11_2420718batch_normalization_11_2420720*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_241928420
.batch_normalization_11/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_5_2420723conv2d_transpose_5_2420725conv2d_transpose_5_2420727*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_24193242,
*conv2d_transpose_5/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_12_2420730batch_normalization_12_2420732batch_normalization_12_2420734batch_normalization_12_2420736*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_241934920
.batch_normalization_12/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_transpose_6_2420739conv2d_transpose_6_2420741conv2d_transpose_6_2420743*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_24193892,
*conv2d_transpose_6/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_13_2420746batch_normalization_13_2420748batch_normalization_13_2420750batch_normalization_13_2420752*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_241941420
.batch_normalization_13/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_transpose_7_2420755conv2d_transpose_7_2420757conv2d_transpose_7_2420759*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_24194542,
*conv2d_transpose_7/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_14_2420762batch_normalization_14_2420764batch_normalization_14_2420766batch_normalization_14_2420768*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_241947920
.batch_normalization_14/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_transpose_8_2420771conv2d_transpose_8_2420773conv2d_transpose_8_2420775*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_24195192,
*conv2d_transpose_8/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_15_2420778batch_normalization_15_2420780batch_normalization_15_2420782batch_normalization_15_2420784*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_241954420
.batch_normalization_15/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv2d_transpose_9_2420787conv2d_transpose_9_2420789*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_24195772,
*conv2d_transpose_9/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2419479

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_15_layer_call_fn_2423381

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_24197452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
??
?:
D__inference_model_1_layer_call_and_return_conditional_losses_2421655

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: B
*conv2d_4_p_re_lu_8_readvariableop_resource:?? ;
-batch_normalization_8_readvariableop_resource: =
/batch_normalization_8_readvariableop_1_resource: L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_5_conv2d_readvariableop_resource: 06
(conv2d_5_biasadd_readvariableop_resource:0B
*conv2d_5_p_re_lu_9_readvariableop_resource:??0;
-batch_normalization_9_readvariableop_resource:0=
/batch_normalization_9_readvariableop_1_resource:0L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_6_conv2d_readvariableop_resource:0@6
(conv2d_6_biasadd_readvariableop_resource:@A
+conv2d_6_p_re_lu_10_readvariableop_resource:@@@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@6
(conv2d_7_biasadd_readvariableop_resource:A
+conv2d_7_p_re_lu_11_readvariableop_resource:  <
.batch_normalization_11_readvariableop_resource:>
0batch_normalization_11_readvariableop_1_resource:M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:K
5conv2d_transpose_5_p_re_lu_12_readvariableop_resource:@@<
.batch_normalization_12_readvariableop_resource:>
0batch_normalization_12_readvariableop_1_resource:M
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_6_biasadd_readvariableop_resource: M
5conv2d_transpose_6_p_re_lu_13_readvariableop_resource:?? <
.batch_normalization_13_readvariableop_resource: >
0batch_normalization_13_readvariableop_1_resource: M
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource: U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:0 @
2conv2d_transpose_7_biasadd_readvariableop_resource:0M
5conv2d_transpose_7_p_re_lu_14_readvariableop_resource:??0<
.batch_normalization_14_readvariableop_resource:0>
0batch_normalization_14_readvariableop_1_resource:0M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:0O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:0U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource:@0@
2conv2d_transpose_8_biasadd_readvariableop_resource:@M
5conv2d_transpose_8_p_re_lu_15_readvariableop_resource:??@<
.batch_normalization_15_readvariableop_resource:@>
0batch_normalization_15_readvariableop_1_resource:@M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_9_biasadd_readvariableop_resource:
identity??%batch_normalization_10/AssignNewValue?'batch_normalization_10/AssignNewValue_1?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?%batch_normalization_11/AssignNewValue?'batch_normalization_11/AssignNewValue_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?%batch_normalization_12/AssignNewValue?'batch_normalization_12/AssignNewValue_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?%batch_normalization_14/AssignNewValue?'batch_normalization_14/AssignNewValue_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?!conv2d_4/p_re_lu_8/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?!conv2d_5/p_re_lu_9/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?"conv2d_6/p_re_lu_10/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?"conv2d_7/p_re_lu_11/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?,conv2d_transpose_5/p_re_lu_12/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?,conv2d_transpose_6/p_re_lu_13/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?,conv2d_transpose_7/p_re_lu_14/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?,conv2d_transpose_8/p_re_lu_15/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/BiasAdd?
conv2d_4/p_re_lu_8/ReluReluconv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/Relu?
!conv2d_4/p_re_lu_8/ReadVariableOpReadVariableOp*conv2d_4_p_re_lu_8_readvariableop_resource*$
_output_shapes
:?? *
dtype02#
!conv2d_4/p_re_lu_8/ReadVariableOp?
conv2d_4/p_re_lu_8/NegNeg)conv2d_4/p_re_lu_8/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
conv2d_4/p_re_lu_8/Neg?
conv2d_4/p_re_lu_8/Neg_1Negconv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/Neg_1?
conv2d_4/p_re_lu_8/Relu_1Reluconv2d_4/p_re_lu_8/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/Relu_1?
conv2d_4/p_re_lu_8/mulMulconv2d_4/p_re_lu_8/Neg:y:0'conv2d_4/p_re_lu_8/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/mul?
conv2d_4/p_re_lu_8/addAddV2%conv2d_4/p_re_lu_8/Relu:activations:0conv2d_4/p_re_lu_8/mul:z:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/add?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/p_re_lu_8/add:z:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_5/BiasAdd?
conv2d_5/p_re_lu_9/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/Relu?
!conv2d_5/p_re_lu_9/ReadVariableOpReadVariableOp*conv2d_5_p_re_lu_9_readvariableop_resource*$
_output_shapes
:??0*
dtype02#
!conv2d_5/p_re_lu_9/ReadVariableOp?
conv2d_5/p_re_lu_9/NegNeg)conv2d_5/p_re_lu_9/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
conv2d_5/p_re_lu_9/Neg?
conv2d_5/p_re_lu_9/Neg_1Negconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/Neg_1?
conv2d_5/p_re_lu_9/Relu_1Reluconv2d_5/p_re_lu_9/Neg_1:y:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/Relu_1?
conv2d_5/p_re_lu_9/mulMulconv2d_5/p_re_lu_9/Neg:y:0'conv2d_5/p_re_lu_9/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/mul?
conv2d_5/p_re_lu_9/addAddV2%conv2d_5/p_re_lu_9/Relu:activations:0conv2d_5/p_re_lu_9/mul:z:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/add?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/p_re_lu_9/add:z:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/BiasAdd?
conv2d_6/p_re_lu_10/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/Relu?
"conv2d_6/p_re_lu_10/ReadVariableOpReadVariableOp+conv2d_6_p_re_lu_10_readvariableop_resource*"
_output_shapes
:@@@*
dtype02$
"conv2d_6/p_re_lu_10/ReadVariableOp?
conv2d_6/p_re_lu_10/NegNeg*conv2d_6/p_re_lu_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2
conv2d_6/p_re_lu_10/Neg?
conv2d_6/p_re_lu_10/Neg_1Negconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/Neg_1?
conv2d_6/p_re_lu_10/Relu_1Reluconv2d_6/p_re_lu_10/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/Relu_1?
conv2d_6/p_re_lu_10/mulMulconv2d_6/p_re_lu_10/Neg:y:0(conv2d_6/p_re_lu_10/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/mul?
conv2d_6/p_re_lu_10/addAddV2&conv2d_6/p_re_lu_10/Relu:activations:0conv2d_6/p_re_lu_10/mul:z:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/add?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_6/p_re_lu_10/add:z:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_10/FusedBatchNormV3?
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue?
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/BiasAdd?
conv2d_7/p_re_lu_11/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/Relu?
"conv2d_7/p_re_lu_11/ReadVariableOpReadVariableOp+conv2d_7_p_re_lu_11_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv2d_7/p_re_lu_11/ReadVariableOp?
conv2d_7/p_re_lu_11/NegNeg*conv2d_7/p_re_lu_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2
conv2d_7/p_re_lu_11/Neg?
conv2d_7/p_re_lu_11/Neg_1Negconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/Neg_1?
conv2d_7/p_re_lu_11/Relu_1Reluconv2d_7/p_re_lu_11/Neg_1:y:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/Relu_1?
conv2d_7/p_re_lu_11/mulMulconv2d_7/p_re_lu_11/Neg:y:0(conv2d_7/p_re_lu_11/Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/mul?
conv2d_7/p_re_lu_11/addAddV2&conv2d_7/p_re_lu_11/Relu:activations:0conv2d_7/p_re_lu_11/mul:z:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/add?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_7/p_re_lu_11/add:z:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_11/FusedBatchNormV3?
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue?
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1?
conv2d_transpose_5/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slicez
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_5/stack/1z
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_5/BiasAdd?
"conv2d_transpose_5/p_re_lu_12/ReluRelu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2$
"conv2d_transpose_5/p_re_lu_12/Relu?
,conv2d_transpose_5/p_re_lu_12/ReadVariableOpReadVariableOp5conv2d_transpose_5_p_re_lu_12_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv2d_transpose_5/p_re_lu_12/ReadVariableOp?
!conv2d_transpose_5/p_re_lu_12/NegNeg4conv2d_transpose_5/p_re_lu_12/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2#
!conv2d_transpose_5/p_re_lu_12/Neg?
#conv2d_transpose_5/p_re_lu_12/Neg_1Neg#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2%
#conv2d_transpose_5/p_re_lu_12/Neg_1?
$conv2d_transpose_5/p_re_lu_12/Relu_1Relu'conv2d_transpose_5/p_re_lu_12/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@2&
$conv2d_transpose_5/p_re_lu_12/Relu_1?
!conv2d_transpose_5/p_re_lu_12/mulMul%conv2d_transpose_5/p_re_lu_12/Neg:y:02conv2d_transpose_5/p_re_lu_12/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2#
!conv2d_transpose_5/p_re_lu_12/mul?
!conv2d_transpose_5/p_re_lu_12/addAddV20conv2d_transpose_5/p_re_lu_12/Relu:activations:0%conv2d_transpose_5/p_re_lu_12/mul:z:0*
T0*/
_output_shapes
:?????????@@2#
!conv2d_transpose_5/p_re_lu_12/add?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_5/p_re_lu_12/add:z:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_12/FusedBatchNormV3?
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue?
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1?
conv2d_transpose_6/ShapeShape+batch_normalization_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slice{
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_6/stack/1{
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose_6/BiasAdd?
"conv2d_transpose_6/p_re_lu_13/ReluRelu#conv2d_transpose_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2$
"conv2d_transpose_6/p_re_lu_13/Relu?
,conv2d_transpose_6/p_re_lu_13/ReadVariableOpReadVariableOp5conv2d_transpose_6_p_re_lu_13_readvariableop_resource*$
_output_shapes
:?? *
dtype02.
,conv2d_transpose_6/p_re_lu_13/ReadVariableOp?
!conv2d_transpose_6/p_re_lu_13/NegNeg4conv2d_transpose_6/p_re_lu_13/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2#
!conv2d_transpose_6/p_re_lu_13/Neg?
#conv2d_transpose_6/p_re_lu_13/Neg_1Neg#conv2d_transpose_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2%
#conv2d_transpose_6/p_re_lu_13/Neg_1?
$conv2d_transpose_6/p_re_lu_13/Relu_1Relu'conv2d_transpose_6/p_re_lu_13/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2&
$conv2d_transpose_6/p_re_lu_13/Relu_1?
!conv2d_transpose_6/p_re_lu_13/mulMul%conv2d_transpose_6/p_re_lu_13/Neg:y:02conv2d_transpose_6/p_re_lu_13/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2#
!conv2d_transpose_6/p_re_lu_13/mul?
!conv2d_transpose_6/p_re_lu_13/addAddV20conv2d_transpose_6/p_re_lu_13/Relu:activations:0%conv2d_transpose_6/p_re_lu_13/mul:z:0*
T0*1
_output_shapes
:??????????? 2#
!conv2d_transpose_6/p_re_lu_13/add?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_6/p_re_lu_13/add:z:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_13/FusedBatchNormV3?
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue?
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1?
conv2d_transpose_7/ShapeShape+batch_normalization_13/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_7/Shape?
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_7/strided_slice/stack?
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_1?
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_2?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_7/strided_slice{
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_7/stack/1{
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_7/stack/2z
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_7/stack/3?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_7/stack?
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_7/strided_slice_1/stack?
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_1?
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_2?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_7/strided_slice_1?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype024
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_13/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2%
#conv2d_transpose_7/conv2d_transpose?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)conv2d_transpose_7/BiasAdd/ReadVariableOp?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_transpose_7/BiasAdd?
"conv2d_transpose_7/p_re_lu_14/ReluRelu#conv2d_transpose_7/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02$
"conv2d_transpose_7/p_re_lu_14/Relu?
,conv2d_transpose_7/p_re_lu_14/ReadVariableOpReadVariableOp5conv2d_transpose_7_p_re_lu_14_readvariableop_resource*$
_output_shapes
:??0*
dtype02.
,conv2d_transpose_7/p_re_lu_14/ReadVariableOp?
!conv2d_transpose_7/p_re_lu_14/NegNeg4conv2d_transpose_7/p_re_lu_14/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02#
!conv2d_transpose_7/p_re_lu_14/Neg?
#conv2d_transpose_7/p_re_lu_14/Neg_1Neg#conv2d_transpose_7/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02%
#conv2d_transpose_7/p_re_lu_14/Neg_1?
$conv2d_transpose_7/p_re_lu_14/Relu_1Relu'conv2d_transpose_7/p_re_lu_14/Neg_1:y:0*
T0*1
_output_shapes
:???????????02&
$conv2d_transpose_7/p_re_lu_14/Relu_1?
!conv2d_transpose_7/p_re_lu_14/mulMul%conv2d_transpose_7/p_re_lu_14/Neg:y:02conv2d_transpose_7/p_re_lu_14/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02#
!conv2d_transpose_7/p_re_lu_14/mul?
!conv2d_transpose_7/p_re_lu_14/addAddV20conv2d_transpose_7/p_re_lu_14/Relu:activations:0%conv2d_transpose_7/p_re_lu_14/mul:z:0*
T0*1
_output_shapes
:???????????02#
!conv2d_transpose_7/p_re_lu_14/add?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
:0*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
:0*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_7/p_re_lu_14/add:z:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_14/FusedBatchNormV3?
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_14/AssignNewValue?
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_14/AssignNewValue_1?
conv2d_transpose_8/ShapeShape+batch_normalization_14/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_8/Shape?
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_8/strided_slice/stack?
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_1?
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_2?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_8/strided_slice{
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_8/stack/1{
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_8/stack/2z
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_8/stack/3?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_8/stack?
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_8/strided_slice_1/stack?
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_1?
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_2?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_8/strided_slice_1?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype024
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2%
#conv2d_transpose_8/conv2d_transpose?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_8/BiasAdd/ReadVariableOp?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_8/BiasAdd?
"conv2d_transpose_8/p_re_lu_15/ReluRelu#conv2d_transpose_8/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2$
"conv2d_transpose_8/p_re_lu_15/Relu?
,conv2d_transpose_8/p_re_lu_15/ReadVariableOpReadVariableOp5conv2d_transpose_8_p_re_lu_15_readvariableop_resource*$
_output_shapes
:??@*
dtype02.
,conv2d_transpose_8/p_re_lu_15/ReadVariableOp?
!conv2d_transpose_8/p_re_lu_15/NegNeg4conv2d_transpose_8/p_re_lu_15/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2#
!conv2d_transpose_8/p_re_lu_15/Neg?
#conv2d_transpose_8/p_re_lu_15/Neg_1Neg#conv2d_transpose_8/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#conv2d_transpose_8/p_re_lu_15/Neg_1?
$conv2d_transpose_8/p_re_lu_15/Relu_1Relu'conv2d_transpose_8/p_re_lu_15/Neg_1:y:0*
T0*1
_output_shapes
:???????????@2&
$conv2d_transpose_8/p_re_lu_15/Relu_1?
!conv2d_transpose_8/p_re_lu_15/mulMul%conv2d_transpose_8/p_re_lu_15/Neg:y:02conv2d_transpose_8/p_re_lu_15/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2#
!conv2d_transpose_8/p_re_lu_15/mul?
!conv2d_transpose_8/p_re_lu_15/addAddV20conv2d_transpose_8/p_re_lu_15/Relu:activations:0%conv2d_transpose_8/p_re_lu_15/mul:z:0*
T0*1
_output_shapes
:???????????@2#
!conv2d_transpose_8/p_re_lu_15/add?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_8/p_re_lu_15/add:z:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
conv2d_transpose_9/ShapeShape+batch_normalization_15/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slice{
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/1{
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/2z
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_15/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_9/BiasAdd?
conv2d_transpose_9/SigmoidSigmoid#conv2d_transpose_9/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_9/Sigmoid?
IdentityIdentityconv2d_transpose_9/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp"^conv2d_4/p_re_lu_8/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp"^conv2d_5/p_re_lu_9/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp#^conv2d_6/p_re_lu_10/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp#^conv2d_7/p_re_lu_11/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp-^conv2d_transpose_5/p_re_lu_12/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp-^conv2d_transpose_6/p_re_lu_13/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp-^conv2d_transpose_7/p_re_lu_14/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp-^conv2d_transpose_8/p_re_lu_15/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2F
!conv2d_4/p_re_lu_8/ReadVariableOp!conv2d_4/p_re_lu_8/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2F
!conv2d_5/p_re_lu_9/ReadVariableOp!conv2d_5/p_re_lu_9/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2H
"conv2d_6/p_re_lu_10/ReadVariableOp"conv2d_6/p_re_lu_10/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2H
"conv2d_7/p_re_lu_11/ReadVariableOp"conv2d_7/p_re_lu_11/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_5/p_re_lu_12/ReadVariableOp,conv2d_transpose_5/p_re_lu_12/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_6/p_re_lu_13/ReadVariableOp,conv2d_transpose_6/p_re_lu_13/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_7/p_re_lu_14/ReadVariableOp,conv2d_transpose_7/p_re_lu_14/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_8/p_re_lu_15/ReadVariableOp,conv2d_transpose_8/p_re_lu_15/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2418927

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423075

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_13_layer_call_fn_2423602

inputs
unknown:?? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_24181602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423057

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422839

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2419745

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2418084

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2417713

inputs-
readvariableop_resource:  
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:  *
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:  2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????  2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_9_layer_call_fn_2423448

inputs!
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_24190252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?.
?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2422768

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: :
"p_re_lu_13_readvariableop_resource:?? 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_13/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
p_re_lu_13/ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
p_re_lu_13/Relu?
p_re_lu_13/ReadVariableOpReadVariableOp"p_re_lu_13_readvariableop_resource*$
_output_shapes
:?? *
dtype02
p_re_lu_13/ReadVariableOpy
p_re_lu_13/NegNeg!p_re_lu_13/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
p_re_lu_13/Neg?
p_re_lu_13/Neg_1NegBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
p_re_lu_13/Neg_1?
p_re_lu_13/Relu_1Relup_re_lu_13/Neg_1:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
p_re_lu_13/Relu_1?
p_re_lu_13/mulMulp_re_lu_13/Neg:y:0p_re_lu_13/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/mul?
p_re_lu_13/addAddV2p_re_lu_13/Relu:activations:0p_re_lu_13/mul:z:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_13/addw
IdentityIdentityp_re_lu_13/add:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_13/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_13/ReadVariableOpp_re_lu_13/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422403

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422639

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_14_layer_call_fn_2423137

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_24186462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2417759

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2419414

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2420077

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?"
?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2419454

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0:
"p_re_lu_14_readvariableop_resource:??0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_14/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :02	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddx
p_re_lu_14/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/Relu?
p_re_lu_14/ReadVariableOpReadVariableOp"p_re_lu_14_readvariableop_resource*$
_output_shapes
:??0*
dtype02
p_re_lu_14/ReadVariableOpy
p_re_lu_14/NegNeg!p_re_lu_14/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
p_re_lu_14/Negy
p_re_lu_14/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/Neg_1?
p_re_lu_14/Relu_1Relup_re_lu_14/Neg_1:y:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/Relu_1?
p_re_lu_14/mulMulp_re_lu_14/Neg:y:0p_re_lu_14/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/mul?
p_re_lu_14/addAddV2p_re_lu_14/Relu:activations:0p_re_lu_14/mul:z:0*
T0*1
_output_shapes
:???????????02
p_re_lu_14/addw
IdentityIdentityp_re_lu_14/add:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_14/ReadVariableOpp_re_lu_14/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_2420656
input_2!
unknown: 
	unknown_0: !
	unknown_1:?? 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: 0
	unknown_7:0!
	unknown_8:??0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@ 

unknown_15:@@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@

unknown_21: 

unknown_22:  

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:

unknown_28: 

unknown_29:@@

unknown_30:

unknown_31:

unknown_32:

unknown_33:$

unknown_34: 

unknown_35: "

unknown_36:?? 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:0 

unknown_42:0"

unknown_43:??0

unknown_44:0

unknown_45:0

unknown_46:0

unknown_47:0$

unknown_48:@0

unknown_49:@"

unknown_50:??@

unknown_51:@

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*L
_read_only_resource_inputs.
,*	
 !$%&'(+,-./234569:*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_24204162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2420133

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_9_layer_call_fn_2422177

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24174712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_5_layer_call_fn_2422603

inputs!
unknown:
	unknown_0:
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_24193242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2417215

inputs/
readvariableop_resource:?? 
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:?? *
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:??????????? 2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2422221

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@8
"p_re_lu_10_readvariableop_resource:@@@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_10/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAddv
p_re_lu_10/ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/Relu?
p_re_lu_10/ReadVariableOpReadVariableOp"p_re_lu_10_readvariableop_resource*"
_output_shapes
:@@@*
dtype02
p_re_lu_10/ReadVariableOpw
p_re_lu_10/NegNeg!p_re_lu_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2
p_re_lu_10/Negw
p_re_lu_10/Neg_1NegBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/Neg_1~
p_re_lu_10/Relu_1Relup_re_lu_10/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/Relu_1?
p_re_lu_10/mulMulp_re_lu_10/Neg:y:0p_re_lu_10/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/mul?
p_re_lu_10/addAddV2p_re_lu_10/Relu:activations:0p_re_lu_10/mul:z:0*
T0*/
_output_shapes
:?????????@@@2
p_re_lu_10/addu
IdentityIdentityp_re_lu_10/add:z:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????0: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp26
p_re_lu_10/ReadVariableOpp_re_lu_10/ReadVariableOp:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_13_layer_call_fn_2422919

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_24183652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422250

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422439

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2418231

inputs/
readvariableop_resource:?? 
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:?? *
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:??????????? 2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+??????????????????????????? : 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422675

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423329

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_14_layer_call_fn_2423163

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_24198002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2423621

inputs/
readvariableop_resource:??0
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??0*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??02
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????02
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422304

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_15_layer_call_fn_2423678

inputs
unknown:??@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_24187222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?<
"__inference__wrapped_model_2417199
input_2I
/model_1_conv2d_4_conv2d_readvariableop_resource: >
0model_1_conv2d_4_biasadd_readvariableop_resource: J
2model_1_conv2d_4_p_re_lu_8_readvariableop_resource:?? C
5model_1_batch_normalization_8_readvariableop_resource: E
7model_1_batch_normalization_8_readvariableop_1_resource: T
Fmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource: I
/model_1_conv2d_5_conv2d_readvariableop_resource: 0>
0model_1_conv2d_5_biasadd_readvariableop_resource:0J
2model_1_conv2d_5_p_re_lu_9_readvariableop_resource:??0C
5model_1_batch_normalization_9_readvariableop_resource:0E
7model_1_batch_normalization_9_readvariableop_1_resource:0T
Fmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:0V
Hmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:0I
/model_1_conv2d_6_conv2d_readvariableop_resource:0@>
0model_1_conv2d_6_biasadd_readvariableop_resource:@I
3model_1_conv2d_6_p_re_lu_10_readvariableop_resource:@@@D
6model_1_batch_normalization_10_readvariableop_resource:@F
8model_1_batch_normalization_10_readvariableop_1_resource:@U
Gmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@W
Imodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@I
/model_1_conv2d_7_conv2d_readvariableop_resource:@>
0model_1_conv2d_7_biasadd_readvariableop_resource:I
3model_1_conv2d_7_p_re_lu_11_readvariableop_resource:  D
6model_1_batch_normalization_11_readvariableop_resource:F
8model_1_batch_normalization_11_readvariableop_1_resource:U
Gmodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:W
Imodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:]
Cmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_5_biasadd_readvariableop_resource:S
=model_1_conv2d_transpose_5_p_re_lu_12_readvariableop_resource:@@D
6model_1_batch_normalization_12_readvariableop_resource:F
8model_1_batch_normalization_12_readvariableop_1_resource:U
Gmodel_1_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:W
Imodel_1_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:]
Cmodel_1_conv2d_transpose_6_conv2d_transpose_readvariableop_resource: H
:model_1_conv2d_transpose_6_biasadd_readvariableop_resource: U
=model_1_conv2d_transpose_6_p_re_lu_13_readvariableop_resource:?? D
6model_1_batch_normalization_13_readvariableop_resource: F
8model_1_batch_normalization_13_readvariableop_1_resource: U
Gmodel_1_batch_normalization_13_fusedbatchnormv3_readvariableop_resource: W
Imodel_1_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource: ]
Cmodel_1_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:0 H
:model_1_conv2d_transpose_7_biasadd_readvariableop_resource:0U
=model_1_conv2d_transpose_7_p_re_lu_14_readvariableop_resource:??0D
6model_1_batch_normalization_14_readvariableop_resource:0F
8model_1_batch_normalization_14_readvariableop_1_resource:0U
Gmodel_1_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:0W
Imodel_1_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:0]
Cmodel_1_conv2d_transpose_8_conv2d_transpose_readvariableop_resource:@0H
:model_1_conv2d_transpose_8_biasadd_readvariableop_resource:@U
=model_1_conv2d_transpose_8_p_re_lu_15_readvariableop_resource:??@D
6model_1_batch_normalization_15_readvariableop_resource:@F
8model_1_batch_normalization_15_readvariableop_1_resource:@U
Gmodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@W
Imodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@]
Cmodel_1_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@H
:model_1_conv2d_transpose_9_biasadd_readvariableop_resource:
identity??>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_10/ReadVariableOp?/model_1/batch_normalization_10/ReadVariableOp_1?>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_11/ReadVariableOp?/model_1/batch_normalization_11/ReadVariableOp_1?>model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_12/ReadVariableOp?/model_1/batch_normalization_12/ReadVariableOp_1?>model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_13/ReadVariableOp?/model_1/batch_normalization_13/ReadVariableOp_1?>model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_14/ReadVariableOp?/model_1/batch_normalization_14/ReadVariableOp_1?>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_15/ReadVariableOp?/model_1/batch_normalization_15/ReadVariableOp_1?=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_8/ReadVariableOp?.model_1/batch_normalization_8/ReadVariableOp_1?=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_9/ReadVariableOp?.model_1/batch_normalization_9/ReadVariableOp_1?'model_1/conv2d_4/BiasAdd/ReadVariableOp?&model_1/conv2d_4/Conv2D/ReadVariableOp?)model_1/conv2d_4/p_re_lu_8/ReadVariableOp?'model_1/conv2d_5/BiasAdd/ReadVariableOp?&model_1/conv2d_5/Conv2D/ReadVariableOp?)model_1/conv2d_5/p_re_lu_9/ReadVariableOp?'model_1/conv2d_6/BiasAdd/ReadVariableOp?&model_1/conv2d_6/Conv2D/ReadVariableOp?*model_1/conv2d_6/p_re_lu_10/ReadVariableOp?'model_1/conv2d_7/BiasAdd/ReadVariableOp?&model_1/conv2d_7/Conv2D/ReadVariableOp?*model_1/conv2d_7/p_re_lu_11/ReadVariableOp?1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?4model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOp?1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?4model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOp?1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?4model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOp?1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?4model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOp?1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
&model_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_1/conv2d_4/Conv2D/ReadVariableOp?
model_1/conv2d_4/Conv2DConv2Dinput_2.model_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_1/conv2d_4/Conv2D?
'model_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_4/BiasAdd/ReadVariableOp?
model_1/conv2d_4/BiasAddBiasAdd model_1/conv2d_4/Conv2D:output:0/model_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model_1/conv2d_4/BiasAdd?
model_1/conv2d_4/p_re_lu_8/ReluRelu!model_1/conv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2!
model_1/conv2d_4/p_re_lu_8/Relu?
)model_1/conv2d_4/p_re_lu_8/ReadVariableOpReadVariableOp2model_1_conv2d_4_p_re_lu_8_readvariableop_resource*$
_output_shapes
:?? *
dtype02+
)model_1/conv2d_4/p_re_lu_8/ReadVariableOp?
model_1/conv2d_4/p_re_lu_8/NegNeg1model_1/conv2d_4/p_re_lu_8/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2 
model_1/conv2d_4/p_re_lu_8/Neg?
 model_1/conv2d_4/p_re_lu_8/Neg_1Neg!model_1/conv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2"
 model_1/conv2d_4/p_re_lu_8/Neg_1?
!model_1/conv2d_4/p_re_lu_8/Relu_1Relu$model_1/conv2d_4/p_re_lu_8/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2#
!model_1/conv2d_4/p_re_lu_8/Relu_1?
model_1/conv2d_4/p_re_lu_8/mulMul"model_1/conv2d_4/p_re_lu_8/Neg:y:0/model_1/conv2d_4/p_re_lu_8/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2 
model_1/conv2d_4/p_re_lu_8/mul?
model_1/conv2d_4/p_re_lu_8/addAddV2-model_1/conv2d_4/p_re_lu_8/Relu:activations:0"model_1/conv2d_4/p_re_lu_8/mul:z:0*
T0*1
_output_shapes
:??????????? 2 
model_1/conv2d_4/p_re_lu_8/add?
,model_1/batch_normalization_8/ReadVariableOpReadVariableOp5model_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_8/ReadVariableOp?
.model_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_8/ReadVariableOp_1?
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_4/p_re_lu_8/add:z:04model_1/batch_normalization_8/ReadVariableOp:value:06model_1/batch_normalization_8/ReadVariableOp_1:value:0Emodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 20
.model_1/batch_normalization_8/FusedBatchNormV3?
&model_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02(
&model_1/conv2d_5/Conv2D/ReadVariableOp?
model_1/conv2d_5/Conv2DConv2D2model_1/batch_normalization_8/FusedBatchNormV3:y:0.model_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
model_1/conv2d_5/Conv2D?
'model_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'model_1/conv2d_5/BiasAdd/ReadVariableOp?
model_1/conv2d_5/BiasAddBiasAdd model_1/conv2d_5/Conv2D:output:0/model_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
model_1/conv2d_5/BiasAdd?
model_1/conv2d_5/p_re_lu_9/ReluRelu!model_1/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02!
model_1/conv2d_5/p_re_lu_9/Relu?
)model_1/conv2d_5/p_re_lu_9/ReadVariableOpReadVariableOp2model_1_conv2d_5_p_re_lu_9_readvariableop_resource*$
_output_shapes
:??0*
dtype02+
)model_1/conv2d_5/p_re_lu_9/ReadVariableOp?
model_1/conv2d_5/p_re_lu_9/NegNeg1model_1/conv2d_5/p_re_lu_9/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02 
model_1/conv2d_5/p_re_lu_9/Neg?
 model_1/conv2d_5/p_re_lu_9/Neg_1Neg!model_1/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02"
 model_1/conv2d_5/p_re_lu_9/Neg_1?
!model_1/conv2d_5/p_re_lu_9/Relu_1Relu$model_1/conv2d_5/p_re_lu_9/Neg_1:y:0*
T0*1
_output_shapes
:???????????02#
!model_1/conv2d_5/p_re_lu_9/Relu_1?
model_1/conv2d_5/p_re_lu_9/mulMul"model_1/conv2d_5/p_re_lu_9/Neg:y:0/model_1/conv2d_5/p_re_lu_9/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02 
model_1/conv2d_5/p_re_lu_9/mul?
model_1/conv2d_5/p_re_lu_9/addAddV2-model_1/conv2d_5/p_re_lu_9/Relu:activations:0"model_1/conv2d_5/p_re_lu_9/mul:z:0*
T0*1
_output_shapes
:???????????02 
model_1/conv2d_5/p_re_lu_9/add?
,model_1/batch_normalization_9/ReadVariableOpReadVariableOp5model_1_batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02.
,model_1/batch_normalization_9/ReadVariableOp?
.model_1/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype020
.model_1/batch_normalization_9/ReadVariableOp_1?
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02?
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02A
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_5/p_re_lu_9/add:z:04model_1/batch_normalization_9/ReadVariableOp:value:06model_1/batch_normalization_9/ReadVariableOp_1:value:0Emodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 20
.model_1/batch_normalization_9/FusedBatchNormV3?
&model_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02(
&model_1/conv2d_6/Conv2D/ReadVariableOp?
model_1/conv2d_6/Conv2DConv2D2model_1/batch_normalization_9/FusedBatchNormV3:y:0.model_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
model_1/conv2d_6/Conv2D?
'model_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv2d_6/BiasAdd/ReadVariableOp?
model_1/conv2d_6/BiasAddBiasAdd model_1/conv2d_6/Conv2D:output:0/model_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
model_1/conv2d_6/BiasAdd?
 model_1/conv2d_6/p_re_lu_10/ReluRelu!model_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2"
 model_1/conv2d_6/p_re_lu_10/Relu?
*model_1/conv2d_6/p_re_lu_10/ReadVariableOpReadVariableOp3model_1_conv2d_6_p_re_lu_10_readvariableop_resource*"
_output_shapes
:@@@*
dtype02,
*model_1/conv2d_6/p_re_lu_10/ReadVariableOp?
model_1/conv2d_6/p_re_lu_10/NegNeg2model_1/conv2d_6/p_re_lu_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2!
model_1/conv2d_6/p_re_lu_10/Neg?
!model_1/conv2d_6/p_re_lu_10/Neg_1Neg!model_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2#
!model_1/conv2d_6/p_re_lu_10/Neg_1?
"model_1/conv2d_6/p_re_lu_10/Relu_1Relu%model_1/conv2d_6/p_re_lu_10/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@@2$
"model_1/conv2d_6/p_re_lu_10/Relu_1?
model_1/conv2d_6/p_re_lu_10/mulMul#model_1/conv2d_6/p_re_lu_10/Neg:y:00model_1/conv2d_6/p_re_lu_10/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2!
model_1/conv2d_6/p_re_lu_10/mul?
model_1/conv2d_6/p_re_lu_10/addAddV2.model_1/conv2d_6/p_re_lu_10/Relu:activations:0#model_1/conv2d_6/p_re_lu_10/mul:z:0*
T0*/
_output_shapes
:?????????@@@2!
model_1/conv2d_6/p_re_lu_10/add?
-model_1/batch_normalization_10/ReadVariableOpReadVariableOp6model_1_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_10/ReadVariableOp?
/model_1/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_10/ReadVariableOp_1?
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3#model_1/conv2d_6/p_re_lu_10/add:z:05model_1/batch_normalization_10/ReadVariableOp:value:07model_1/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_10/FusedBatchNormV3?
&model_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model_1/conv2d_7/Conv2D/ReadVariableOp?
model_1/conv2d_7/Conv2DConv2D3model_1/batch_normalization_10/FusedBatchNormV3:y:0.model_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_1/conv2d_7/Conv2D?
'model_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_7/BiasAdd/ReadVariableOp?
model_1/conv2d_7/BiasAddBiasAdd model_1/conv2d_7/Conv2D:output:0/model_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_1/conv2d_7/BiasAdd?
 model_1/conv2d_7/p_re_lu_11/ReluRelu!model_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2"
 model_1/conv2d_7/p_re_lu_11/Relu?
*model_1/conv2d_7/p_re_lu_11/ReadVariableOpReadVariableOp3model_1_conv2d_7_p_re_lu_11_readvariableop_resource*"
_output_shapes
:  *
dtype02,
*model_1/conv2d_7/p_re_lu_11/ReadVariableOp?
model_1/conv2d_7/p_re_lu_11/NegNeg2model_1/conv2d_7/p_re_lu_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
model_1/conv2d_7/p_re_lu_11/Neg?
!model_1/conv2d_7/p_re_lu_11/Neg_1Neg!model_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2#
!model_1/conv2d_7/p_re_lu_11/Neg_1?
"model_1/conv2d_7/p_re_lu_11/Relu_1Relu%model_1/conv2d_7/p_re_lu_11/Neg_1:y:0*
T0*/
_output_shapes
:?????????  2$
"model_1/conv2d_7/p_re_lu_11/Relu_1?
model_1/conv2d_7/p_re_lu_11/mulMul#model_1/conv2d_7/p_re_lu_11/Neg:y:00model_1/conv2d_7/p_re_lu_11/Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2!
model_1/conv2d_7/p_re_lu_11/mul?
model_1/conv2d_7/p_re_lu_11/addAddV2.model_1/conv2d_7/p_re_lu_11/Relu:activations:0#model_1/conv2d_7/p_re_lu_11/mul:z:0*
T0*/
_output_shapes
:?????????  2!
model_1/conv2d_7/p_re_lu_11/add?
-model_1/batch_normalization_11/ReadVariableOpReadVariableOp6model_1_batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_1/batch_normalization_11/ReadVariableOp?
/model_1/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_1/batch_normalization_11/ReadVariableOp_1?
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3#model_1/conv2d_7/p_re_lu_11/add:z:05model_1/batch_normalization_11/ReadVariableOp:value:07model_1/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_11/FusedBatchNormV3?
 model_1/conv2d_transpose_5/ShapeShape3model_1/batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_5/Shape?
.model_1/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_5/strided_slice/stack?
0model_1/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_5/strided_slice/stack_1?
0model_1/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_5/strided_slice/stack_2?
(model_1/conv2d_transpose_5/strided_sliceStridedSlice)model_1/conv2d_transpose_5/Shape:output:07model_1/conv2d_transpose_5/strided_slice/stack:output:09model_1/conv2d_transpose_5/strided_slice/stack_1:output:09model_1/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_5/strided_slice?
"model_1/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2$
"model_1/conv2d_transpose_5/stack/1?
"model_1/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2$
"model_1/conv2d_transpose_5/stack/2?
"model_1/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_5/stack/3?
 model_1/conv2d_transpose_5/stackPack1model_1/conv2d_transpose_5/strided_slice:output:0+model_1/conv2d_transpose_5/stack/1:output:0+model_1/conv2d_transpose_5/stack/2:output:0+model_1/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_5/stack?
0model_1/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_5/strided_slice_1/stack?
2model_1/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_5/strided_slice_1/stack_1?
2model_1/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_5/strided_slice_1/stack_2?
*model_1/conv2d_transpose_5/strided_slice_1StridedSlice)model_1/conv2d_transpose_5/stack:output:09model_1/conv2d_transpose_5/strided_slice_1/stack:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_5/strided_slice_1?
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02<
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_5/stack:output:0Bmodel_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:03model_1/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_5/conv2d_transpose?
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_5/BiasAddBiasAdd4model_1/conv2d_transpose_5/conv2d_transpose:output:09model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2$
"model_1/conv2d_transpose_5/BiasAdd?
*model_1/conv2d_transpose_5/p_re_lu_12/ReluRelu+model_1/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2,
*model_1/conv2d_transpose_5/p_re_lu_12/Relu?
4model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOpReadVariableOp=model_1_conv2d_transpose_5_p_re_lu_12_readvariableop_resource*"
_output_shapes
:@@*
dtype026
4model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOp?
)model_1/conv2d_transpose_5/p_re_lu_12/NegNeg<model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)model_1/conv2d_transpose_5/p_re_lu_12/Neg?
+model_1/conv2d_transpose_5/p_re_lu_12/Neg_1Neg+model_1/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2-
+model_1/conv2d_transpose_5/p_re_lu_12/Neg_1?
,model_1/conv2d_transpose_5/p_re_lu_12/Relu_1Relu/model_1/conv2d_transpose_5/p_re_lu_12/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@2.
,model_1/conv2d_transpose_5/p_re_lu_12/Relu_1?
)model_1/conv2d_transpose_5/p_re_lu_12/mulMul-model_1/conv2d_transpose_5/p_re_lu_12/Neg:y:0:model_1/conv2d_transpose_5/p_re_lu_12/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2+
)model_1/conv2d_transpose_5/p_re_lu_12/mul?
)model_1/conv2d_transpose_5/p_re_lu_12/addAddV28model_1/conv2d_transpose_5/p_re_lu_12/Relu:activations:0-model_1/conv2d_transpose_5/p_re_lu_12/mul:z:0*
T0*/
_output_shapes
:?????????@@2+
)model_1/conv2d_transpose_5/p_re_lu_12/add?
-model_1/batch_normalization_12/ReadVariableOpReadVariableOp6model_1_batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_1/batch_normalization_12/ReadVariableOp?
/model_1/batch_normalization_12/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_1/batch_normalization_12/ReadVariableOp_1?
>model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3-model_1/conv2d_transpose_5/p_re_lu_12/add:z:05model_1/batch_normalization_12/ReadVariableOp:value:07model_1/batch_normalization_12/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_12/FusedBatchNormV3?
 model_1/conv2d_transpose_6/ShapeShape3model_1/batch_normalization_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_6/Shape?
.model_1/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_6/strided_slice/stack?
0model_1/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_6/strided_slice/stack_1?
0model_1/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_6/strided_slice/stack_2?
(model_1/conv2d_transpose_6/strided_sliceStridedSlice)model_1/conv2d_transpose_6/Shape:output:07model_1/conv2d_transpose_6/strided_slice/stack:output:09model_1/conv2d_transpose_6/strided_slice/stack_1:output:09model_1/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_6/strided_slice?
"model_1/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_6/stack/1?
"model_1/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_6/stack/2?
"model_1/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"model_1/conv2d_transpose_6/stack/3?
 model_1/conv2d_transpose_6/stackPack1model_1/conv2d_transpose_6/strided_slice:output:0+model_1/conv2d_transpose_6/stack/1:output:0+model_1/conv2d_transpose_6/stack/2:output:0+model_1/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_6/stack?
0model_1/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_6/strided_slice_1/stack?
2model_1/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_6/strided_slice_1/stack_1?
2model_1/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_6/strided_slice_1/stack_2?
*model_1/conv2d_transpose_6/strided_slice_1StridedSlice)model_1/conv2d_transpose_6/stack:output:09model_1/conv2d_transpose_6/strided_slice_1/stack:output:0;model_1/conv2d_transpose_6/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_6/strided_slice_1?
:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_6/stack:output:0Bmodel_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:03model_1/batch_normalization_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2-
+model_1/conv2d_transpose_6/conv2d_transpose?
1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_6/BiasAddBiasAdd4model_1/conv2d_transpose_6/conv2d_transpose:output:09model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2$
"model_1/conv2d_transpose_6/BiasAdd?
*model_1/conv2d_transpose_6/p_re_lu_13/ReluRelu+model_1/conv2d_transpose_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2,
*model_1/conv2d_transpose_6/p_re_lu_13/Relu?
4model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOpReadVariableOp=model_1_conv2d_transpose_6_p_re_lu_13_readvariableop_resource*$
_output_shapes
:?? *
dtype026
4model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOp?
)model_1/conv2d_transpose_6/p_re_lu_13/NegNeg<model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2+
)model_1/conv2d_transpose_6/p_re_lu_13/Neg?
+model_1/conv2d_transpose_6/p_re_lu_13/Neg_1Neg+model_1/conv2d_transpose_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2-
+model_1/conv2d_transpose_6/p_re_lu_13/Neg_1?
,model_1/conv2d_transpose_6/p_re_lu_13/Relu_1Relu/model_1/conv2d_transpose_6/p_re_lu_13/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2.
,model_1/conv2d_transpose_6/p_re_lu_13/Relu_1?
)model_1/conv2d_transpose_6/p_re_lu_13/mulMul-model_1/conv2d_transpose_6/p_re_lu_13/Neg:y:0:model_1/conv2d_transpose_6/p_re_lu_13/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2+
)model_1/conv2d_transpose_6/p_re_lu_13/mul?
)model_1/conv2d_transpose_6/p_re_lu_13/addAddV28model_1/conv2d_transpose_6/p_re_lu_13/Relu:activations:0-model_1/conv2d_transpose_6/p_re_lu_13/mul:z:0*
T0*1
_output_shapes
:??????????? 2+
)model_1/conv2d_transpose_6/p_re_lu_13/add?
-model_1/batch_normalization_13/ReadVariableOpReadVariableOp6model_1_batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_1/batch_normalization_13/ReadVariableOp?
/model_1/batch_normalization_13/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_1/batch_normalization_13/ReadVariableOp_1?
>model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-model_1/conv2d_transpose_6/p_re_lu_13/add:z:05model_1/batch_normalization_13/ReadVariableOp:value:07model_1/batch_normalization_13/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_13/FusedBatchNormV3?
 model_1/conv2d_transpose_7/ShapeShape3model_1/batch_normalization_13/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_7/Shape?
.model_1/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_7/strided_slice/stack?
0model_1/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_7/strided_slice/stack_1?
0model_1/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_7/strided_slice/stack_2?
(model_1/conv2d_transpose_7/strided_sliceStridedSlice)model_1/conv2d_transpose_7/Shape:output:07model_1/conv2d_transpose_7/strided_slice/stack:output:09model_1/conv2d_transpose_7/strided_slice/stack_1:output:09model_1/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_7/strided_slice?
"model_1/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_7/stack/1?
"model_1/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_7/stack/2?
"model_1/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02$
"model_1/conv2d_transpose_7/stack/3?
 model_1/conv2d_transpose_7/stackPack1model_1/conv2d_transpose_7/strided_slice:output:0+model_1/conv2d_transpose_7/stack/1:output:0+model_1/conv2d_transpose_7/stack/2:output:0+model_1/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_7/stack?
0model_1/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_7/strided_slice_1/stack?
2model_1/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_7/strided_slice_1/stack_1?
2model_1/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_7/strided_slice_1/stack_2?
*model_1/conv2d_transpose_7/strided_slice_1StridedSlice)model_1/conv2d_transpose_7/stack:output:09model_1/conv2d_transpose_7/strided_slice_1/stack:output:0;model_1/conv2d_transpose_7/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_7/strided_slice_1?
:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype02<
:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_7/stack:output:0Bmodel_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:03model_1/batch_normalization_13/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_7/conv2d_transpose?
1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_7/BiasAddBiasAdd4model_1/conv2d_transpose_7/conv2d_transpose:output:09model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02$
"model_1/conv2d_transpose_7/BiasAdd?
*model_1/conv2d_transpose_7/p_re_lu_14/ReluRelu+model_1/conv2d_transpose_7/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02,
*model_1/conv2d_transpose_7/p_re_lu_14/Relu?
4model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOpReadVariableOp=model_1_conv2d_transpose_7_p_re_lu_14_readvariableop_resource*$
_output_shapes
:??0*
dtype026
4model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOp?
)model_1/conv2d_transpose_7/p_re_lu_14/NegNeg<model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02+
)model_1/conv2d_transpose_7/p_re_lu_14/Neg?
+model_1/conv2d_transpose_7/p_re_lu_14/Neg_1Neg+model_1/conv2d_transpose_7/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02-
+model_1/conv2d_transpose_7/p_re_lu_14/Neg_1?
,model_1/conv2d_transpose_7/p_re_lu_14/Relu_1Relu/model_1/conv2d_transpose_7/p_re_lu_14/Neg_1:y:0*
T0*1
_output_shapes
:???????????02.
,model_1/conv2d_transpose_7/p_re_lu_14/Relu_1?
)model_1/conv2d_transpose_7/p_re_lu_14/mulMul-model_1/conv2d_transpose_7/p_re_lu_14/Neg:y:0:model_1/conv2d_transpose_7/p_re_lu_14/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02+
)model_1/conv2d_transpose_7/p_re_lu_14/mul?
)model_1/conv2d_transpose_7/p_re_lu_14/addAddV28model_1/conv2d_transpose_7/p_re_lu_14/Relu:activations:0-model_1/conv2d_transpose_7/p_re_lu_14/mul:z:0*
T0*1
_output_shapes
:???????????02+
)model_1/conv2d_transpose_7/p_re_lu_14/add?
-model_1/batch_normalization_14/ReadVariableOpReadVariableOp6model_1_batch_normalization_14_readvariableop_resource*
_output_shapes
:0*
dtype02/
-model_1/batch_normalization_14/ReadVariableOp?
/model_1/batch_normalization_14/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/model_1/batch_normalization_14/ReadVariableOp_1?
>model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3-model_1/conv2d_transpose_7/p_re_lu_14/add:z:05model_1/batch_normalization_14/ReadVariableOp:value:07model_1/batch_normalization_14/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_14/FusedBatchNormV3?
 model_1/conv2d_transpose_8/ShapeShape3model_1/batch_normalization_14/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_8/Shape?
.model_1/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_8/strided_slice/stack?
0model_1/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_8/strided_slice/stack_1?
0model_1/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_8/strided_slice/stack_2?
(model_1/conv2d_transpose_8/strided_sliceStridedSlice)model_1/conv2d_transpose_8/Shape:output:07model_1/conv2d_transpose_8/strided_slice/stack:output:09model_1/conv2d_transpose_8/strided_slice/stack_1:output:09model_1/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_8/strided_slice?
"model_1/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_8/stack/1?
"model_1/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_8/stack/2?
"model_1/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"model_1/conv2d_transpose_8/stack/3?
 model_1/conv2d_transpose_8/stackPack1model_1/conv2d_transpose_8/strided_slice:output:0+model_1/conv2d_transpose_8/stack/1:output:0+model_1/conv2d_transpose_8/stack/2:output:0+model_1/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_8/stack?
0model_1/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_8/strided_slice_1/stack?
2model_1/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_8/strided_slice_1/stack_1?
2model_1/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_8/strided_slice_1/stack_2?
*model_1/conv2d_transpose_8/strided_slice_1StridedSlice)model_1/conv2d_transpose_8/stack:output:09model_1/conv2d_transpose_8/strided_slice_1/stack:output:0;model_1/conv2d_transpose_8/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_8/strided_slice_1?
:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype02<
:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_8/stack:output:0Bmodel_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:03model_1/batch_normalization_14/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_8/conv2d_transpose?
1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_8/BiasAddBiasAdd4model_1/conv2d_transpose_8/conv2d_transpose:output:09model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2$
"model_1/conv2d_transpose_8/BiasAdd?
*model_1/conv2d_transpose_8/p_re_lu_15/ReluRelu+model_1/conv2d_transpose_8/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2,
*model_1/conv2d_transpose_8/p_re_lu_15/Relu?
4model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOpReadVariableOp=model_1_conv2d_transpose_8_p_re_lu_15_readvariableop_resource*$
_output_shapes
:??@*
dtype026
4model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOp?
)model_1/conv2d_transpose_8/p_re_lu_15/NegNeg<model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2+
)model_1/conv2d_transpose_8/p_re_lu_15/Neg?
+model_1/conv2d_transpose_8/p_re_lu_15/Neg_1Neg+model_1/conv2d_transpose_8/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2-
+model_1/conv2d_transpose_8/p_re_lu_15/Neg_1?
,model_1/conv2d_transpose_8/p_re_lu_15/Relu_1Relu/model_1/conv2d_transpose_8/p_re_lu_15/Neg_1:y:0*
T0*1
_output_shapes
:???????????@2.
,model_1/conv2d_transpose_8/p_re_lu_15/Relu_1?
)model_1/conv2d_transpose_8/p_re_lu_15/mulMul-model_1/conv2d_transpose_8/p_re_lu_15/Neg:y:0:model_1/conv2d_transpose_8/p_re_lu_15/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2+
)model_1/conv2d_transpose_8/p_re_lu_15/mul?
)model_1/conv2d_transpose_8/p_re_lu_15/addAddV28model_1/conv2d_transpose_8/p_re_lu_15/Relu:activations:0-model_1/conv2d_transpose_8/p_re_lu_15/mul:z:0*
T0*1
_output_shapes
:???????????@2+
)model_1/conv2d_transpose_8/p_re_lu_15/add?
-model_1/batch_normalization_15/ReadVariableOpReadVariableOp6model_1_batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_15/ReadVariableOp?
/model_1/batch_normalization_15/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_15/ReadVariableOp_1?
>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3-model_1/conv2d_transpose_8/p_re_lu_15/add:z:05model_1/batch_normalization_15/ReadVariableOp:value:07model_1/batch_normalization_15/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_15/FusedBatchNormV3?
 model_1/conv2d_transpose_9/ShapeShape3model_1/batch_normalization_15/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_9/Shape?
.model_1/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_9/strided_slice/stack?
0model_1/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_9/strided_slice/stack_1?
0model_1/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_9/strided_slice/stack_2?
(model_1/conv2d_transpose_9/strided_sliceStridedSlice)model_1/conv2d_transpose_9/Shape:output:07model_1/conv2d_transpose_9/strided_slice/stack:output:09model_1/conv2d_transpose_9/strided_slice/stack_1:output:09model_1/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_9/strided_slice?
"model_1/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_9/stack/1?
"model_1/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_9/stack/2?
"model_1/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_9/stack/3?
 model_1/conv2d_transpose_9/stackPack1model_1/conv2d_transpose_9/strided_slice:output:0+model_1/conv2d_transpose_9/stack/1:output:0+model_1/conv2d_transpose_9/stack/2:output:0+model_1/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_9/stack?
0model_1/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_9/strided_slice_1/stack?
2model_1/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_9/strided_slice_1/stack_1?
2model_1/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_9/strided_slice_1/stack_2?
*model_1/conv2d_transpose_9/strided_slice_1StridedSlice)model_1/conv2d_transpose_9/stack:output:09model_1/conv2d_transpose_9/strided_slice_1/stack:output:0;model_1/conv2d_transpose_9/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_9/strided_slice_1?
:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_9/stack:output:0Bmodel_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:03model_1/batch_normalization_15/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_9/conv2d_transpose?
1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_9/BiasAddBiasAdd4model_1/conv2d_transpose_9/conv2d_transpose:output:09model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2$
"model_1/conv2d_transpose_9/BiasAdd?
"model_1/conv2d_transpose_9/SigmoidSigmoid+model_1/conv2d_transpose_9/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2$
"model_1/conv2d_transpose_9/Sigmoid?
IdentityIdentity&model_1/conv2d_transpose_9/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp?^model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_10/ReadVariableOp0^model_1/batch_normalization_10/ReadVariableOp_1?^model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_11/ReadVariableOp0^model_1/batch_normalization_11/ReadVariableOp_1?^model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_12/ReadVariableOp0^model_1/batch_normalization_12/ReadVariableOp_1?^model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_13/ReadVariableOp0^model_1/batch_normalization_13/ReadVariableOp_1?^model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_14/ReadVariableOp0^model_1/batch_normalization_14/ReadVariableOp_1?^model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_15/ReadVariableOp0^model_1/batch_normalization_15/ReadVariableOp_1>^model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_8/ReadVariableOp/^model_1/batch_normalization_8/ReadVariableOp_1>^model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_9/ReadVariableOp/^model_1/batch_normalization_9/ReadVariableOp_1(^model_1/conv2d_4/BiasAdd/ReadVariableOp'^model_1/conv2d_4/Conv2D/ReadVariableOp*^model_1/conv2d_4/p_re_lu_8/ReadVariableOp(^model_1/conv2d_5/BiasAdd/ReadVariableOp'^model_1/conv2d_5/Conv2D/ReadVariableOp*^model_1/conv2d_5/p_re_lu_9/ReadVariableOp(^model_1/conv2d_6/BiasAdd/ReadVariableOp'^model_1/conv2d_6/Conv2D/ReadVariableOp+^model_1/conv2d_6/p_re_lu_10/ReadVariableOp(^model_1/conv2d_7/BiasAdd/ReadVariableOp'^model_1/conv2d_7/Conv2D/ReadVariableOp+^model_1/conv2d_7/p_re_lu_11/ReadVariableOp2^model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp5^model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOp2^model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp5^model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOp2^model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp5^model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOp2^model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp5^model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOp2^model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_10/ReadVariableOp-model_1/batch_normalization_10/ReadVariableOp2b
/model_1/batch_normalization_10/ReadVariableOp_1/model_1/batch_normalization_10/ReadVariableOp_12?
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_11/ReadVariableOp-model_1/batch_normalization_11/ReadVariableOp2b
/model_1/batch_normalization_11/ReadVariableOp_1/model_1/batch_normalization_11/ReadVariableOp_12?
>model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_12/ReadVariableOp-model_1/batch_normalization_12/ReadVariableOp2b
/model_1/batch_normalization_12/ReadVariableOp_1/model_1/batch_normalization_12/ReadVariableOp_12?
>model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_13/ReadVariableOp-model_1/batch_normalization_13/ReadVariableOp2b
/model_1/batch_normalization_13/ReadVariableOp_1/model_1/batch_normalization_13/ReadVariableOp_12?
>model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_14/ReadVariableOp-model_1/batch_normalization_14/ReadVariableOp2b
/model_1/batch_normalization_14/ReadVariableOp_1/model_1/batch_normalization_14/ReadVariableOp_12?
>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_15/ReadVariableOp-model_1/batch_normalization_15/ReadVariableOp2b
/model_1/batch_normalization_15/ReadVariableOp_1/model_1/batch_normalization_15/ReadVariableOp_12~
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_8/ReadVariableOp,model_1/batch_normalization_8/ReadVariableOp2`
.model_1/batch_normalization_8/ReadVariableOp_1.model_1/batch_normalization_8/ReadVariableOp_12~
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_9/ReadVariableOp,model_1/batch_normalization_9/ReadVariableOp2`
.model_1/batch_normalization_9/ReadVariableOp_1.model_1/batch_normalization_9/ReadVariableOp_12R
'model_1/conv2d_4/BiasAdd/ReadVariableOp'model_1/conv2d_4/BiasAdd/ReadVariableOp2P
&model_1/conv2d_4/Conv2D/ReadVariableOp&model_1/conv2d_4/Conv2D/ReadVariableOp2V
)model_1/conv2d_4/p_re_lu_8/ReadVariableOp)model_1/conv2d_4/p_re_lu_8/ReadVariableOp2R
'model_1/conv2d_5/BiasAdd/ReadVariableOp'model_1/conv2d_5/BiasAdd/ReadVariableOp2P
&model_1/conv2d_5/Conv2D/ReadVariableOp&model_1/conv2d_5/Conv2D/ReadVariableOp2V
)model_1/conv2d_5/p_re_lu_9/ReadVariableOp)model_1/conv2d_5/p_re_lu_9/ReadVariableOp2R
'model_1/conv2d_6/BiasAdd/ReadVariableOp'model_1/conv2d_6/BiasAdd/ReadVariableOp2P
&model_1/conv2d_6/Conv2D/ReadVariableOp&model_1/conv2d_6/Conv2D/ReadVariableOp2X
*model_1/conv2d_6/p_re_lu_10/ReadVariableOp*model_1/conv2d_6/p_re_lu_10/ReadVariableOp2R
'model_1/conv2d_7/BiasAdd/ReadVariableOp'model_1/conv2d_7/BiasAdd/ReadVariableOp2P
&model_1/conv2d_7/Conv2D/ReadVariableOp&model_1/conv2d_7/Conv2D/ReadVariableOp2X
*model_1/conv2d_7/p_re_lu_11/ReadVariableOp*model_1/conv2d_7/p_re_lu_11/ReadVariableOp2f
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2l
4model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOp4model_1/conv2d_transpose_5/p_re_lu_12/ReadVariableOp2f
1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2l
4model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOp4model_1/conv2d_transpose_6/p_re_lu_13/ReadVariableOp2f
1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2l
4model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOp4model_1/conv2d_transpose_7/p_re_lu_14/ReadVariableOp2f
1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2l
4model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOp4model_1/conv2d_transpose_8/p_re_lu_15/ReadVariableOp2f
1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2419231

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_10_layer_call_fn_2423514

inputs
unknown:@@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_24175472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2419910

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421944

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2417950

inputs-
readvariableop_resource:@@
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:@@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????@@2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_12_layer_call_fn_2422714

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_24193492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422657

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_15_layer_call_fn_2423368

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_24195442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422097

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_11_layer_call_fn_2422496

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_24192842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2417593

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2418321

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2418441

inputs/
readvariableop_resource:??0
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??0*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??02
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????02
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_15_layer_call_fn_2423685

inputs
unknown:??@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_24187932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2419178

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_2422011

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24172612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?-
#__inference__traced_restore_2424132
file_prefix:
 assignvariableop_conv2d_4_kernel: .
 assignvariableop_1_conv2d_4_bias: <
.assignvariableop_2_batch_normalization_8_gamma: ;
-assignvariableop_3_batch_normalization_8_beta: B
4assignvariableop_4_batch_normalization_8_moving_mean: F
8assignvariableop_5_batch_normalization_8_moving_variance: <
"assignvariableop_6_conv2d_5_kernel: 0.
 assignvariableop_7_conv2d_5_bias:0<
.assignvariableop_8_batch_normalization_9_gamma:0;
-assignvariableop_9_batch_normalization_9_beta:0C
5assignvariableop_10_batch_normalization_9_moving_mean:0G
9assignvariableop_11_batch_normalization_9_moving_variance:0=
#assignvariableop_12_conv2d_6_kernel:0@/
!assignvariableop_13_conv2d_6_bias:@>
0assignvariableop_14_batch_normalization_10_gamma:@=
/assignvariableop_15_batch_normalization_10_beta:@D
6assignvariableop_16_batch_normalization_10_moving_mean:@H
:assignvariableop_17_batch_normalization_10_moving_variance:@=
#assignvariableop_18_conv2d_7_kernel:@/
!assignvariableop_19_conv2d_7_bias:>
0assignvariableop_20_batch_normalization_11_gamma:=
/assignvariableop_21_batch_normalization_11_beta:D
6assignvariableop_22_batch_normalization_11_moving_mean:H
:assignvariableop_23_batch_normalization_11_moving_variance:G
-assignvariableop_24_conv2d_transpose_5_kernel:9
+assignvariableop_25_conv2d_transpose_5_bias:>
0assignvariableop_26_batch_normalization_12_gamma:=
/assignvariableop_27_batch_normalization_12_beta:D
6assignvariableop_28_batch_normalization_12_moving_mean:H
:assignvariableop_29_batch_normalization_12_moving_variance:G
-assignvariableop_30_conv2d_transpose_6_kernel: 9
+assignvariableop_31_conv2d_transpose_6_bias: >
0assignvariableop_32_batch_normalization_13_gamma: =
/assignvariableop_33_batch_normalization_13_beta: D
6assignvariableop_34_batch_normalization_13_moving_mean: H
:assignvariableop_35_batch_normalization_13_moving_variance: G
-assignvariableop_36_conv2d_transpose_7_kernel:0 9
+assignvariableop_37_conv2d_transpose_7_bias:0>
0assignvariableop_38_batch_normalization_14_gamma:0=
/assignvariableop_39_batch_normalization_14_beta:0D
6assignvariableop_40_batch_normalization_14_moving_mean:0H
:assignvariableop_41_batch_normalization_14_moving_variance:0G
-assignvariableop_42_conv2d_transpose_8_kernel:@09
+assignvariableop_43_conv2d_transpose_8_bias:@>
0assignvariableop_44_batch_normalization_15_gamma:@=
/assignvariableop_45_batch_normalization_15_beta:@D
6assignvariableop_46_batch_normalization_15_moving_mean:@H
:assignvariableop_47_batch_normalization_15_moving_variance:@G
-assignvariableop_48_conv2d_transpose_9_kernel:@9
+assignvariableop_49_conv2d_transpose_9_bias:&
assignvariableop_50_sgd_iter:	 '
assignvariableop_51_sgd_decay: *
 assignvariableop_52_sgd_momentum: D
,assignvariableop_53_conv2d_4_p_re_lu_8_alpha:?? D
,assignvariableop_54_conv2d_5_p_re_lu_9_alpha:??0C
-assignvariableop_55_conv2d_6_p_re_lu_10_alpha:@@@C
-assignvariableop_56_conv2d_7_p_re_lu_11_alpha:  M
7assignvariableop_57_conv2d_transpose_5_p_re_lu_12_alpha:@@O
7assignvariableop_58_conv2d_transpose_6_p_re_lu_13_alpha:?? O
7assignvariableop_59_conv2d_transpose_7_p_re_lu_14_alpha:??0O
7assignvariableop_60_conv2d_transpose_8_p_re_lu_15_alpha:??@#
assignvariableop_61_total: #
assignvariableop_62_count: %
assignvariableop_63_total_1: %
assignvariableop_64_count_1: %
assignvariableop_65_total_2: %
assignvariableop_66_count_2: %
assignvariableop_67_total_3: %
assignvariableop_68_count_3: 
identity_70??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/37/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_8_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_8_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_8_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_8_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_9_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_9_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_9_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_9_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_10_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_10_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_10_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_10_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_11_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_11_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_11_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_11_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_conv2d_transpose_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_conv2d_transpose_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_12_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_12_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_12_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_12_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_conv2d_transpose_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_conv2d_transpose_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_13_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_13_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_13_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_13_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp-assignvariableop_36_conv2d_transpose_7_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_conv2d_transpose_7_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_14_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_14_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_14_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_14_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp-assignvariableop_42_conv2d_transpose_8_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_conv2d_transpose_8_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_15_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_15_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_15_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_15_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp-assignvariableop_48_conv2d_transpose_9_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_conv2d_transpose_9_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_sgd_iterIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_sgd_decayIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp assignvariableop_52_sgd_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_conv2d_4_p_re_lu_8_alphaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp,assignvariableop_54_conv2d_5_p_re_lu_9_alphaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp-assignvariableop_55_conv2d_6_p_re_lu_10_alphaIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp-assignvariableop_56_conv2d_7_p_re_lu_11_alphaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_conv2d_transpose_5_p_re_lu_12_alphaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp7assignvariableop_58_conv2d_transpose_6_p_re_lu_13_alphaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_conv2d_transpose_7_p_re_lu_14_alphaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp7assignvariableop_60_conv2d_transpose_8_p_re_lu_15_alphaIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpassignvariableop_61_totalIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpassignvariableop_62_countIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpassignvariableop_63_total_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_1Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpassignvariableop_65_total_2Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpassignvariableop_66_count_2Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpassignvariableop_67_total_3Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpassignvariableop_68_count_3Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69f
Identity_70IdentityIdentity_69:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_70?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_70Identity_70:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?"
?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2423235

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@:
"p_re_lu_15_readvariableop_resource:??@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_15/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddx
p_re_lu_15/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/Relu?
p_re_lu_15/ReadVariableOpReadVariableOp"p_re_lu_15_readvariableop_resource*$
_output_shapes
:??@*
dtype02
p_re_lu_15/ReadVariableOpy
p_re_lu_15/NegNeg!p_re_lu_15/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
p_re_lu_15/Negy
p_re_lu_15/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/Neg_1?
p_re_lu_15/Relu_1Relup_re_lu_15/Neg_1:y:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/Relu_1?
p_re_lu_15/mulMulp_re_lu_15/Neg:y:0p_re_lu_15/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/mul?
p_re_lu_15/addAddV2p_re_lu_15/Relu:activations:0p_re_lu_15/mul:z:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/addw
IdentityIdentityp_re_lu_15/add:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????0: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_15/ReadVariableOpp_re_lu_15/ReadVariableOp:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422133

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422286

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2419349

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
4__inference_conv2d_transpose_6_layer_call_fn_2422810

inputs!
unknown: 
	unknown_0: !
	unknown_1:?? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_24182362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2422068

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:09
!p_re_lu_9_readvariableop_resource:??0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_9/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddv
p_re_lu_9/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/Relu?
p_re_lu_9/ReadVariableOpReadVariableOp!p_re_lu_9_readvariableop_resource*$
_output_shapes
:??0*
dtype02
p_re_lu_9/ReadVariableOpv
p_re_lu_9/NegNeg p_re_lu_9/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
p_re_lu_9/Negw
p_re_lu_9/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/Neg_1}
p_re_lu_9/Relu_1Relup_re_lu_9/Neg_1:y:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/Relu_1?
p_re_lu_9/mulMulp_re_lu_9/Neg:y:0p_re_lu_9/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/mul?
p_re_lu_9/addAddV2p_re_lu_9/Relu:activations:0p_re_lu_9/mul:z:0*
T0*1
_output_shapes
:???????????02
p_re_lu_9/addv
IdentityIdentityp_re_lu_9/add:z:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp24
p_re_lu_9/ReadVariableOpp_re_lu_9/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?z
?
D__inference_model_1_layer_call_and_return_conditional_losses_2420416

inputs*
conv2d_4_2420282: 
conv2d_4_2420284: (
conv2d_4_2420286:?? +
batch_normalization_8_2420289: +
batch_normalization_8_2420291: +
batch_normalization_8_2420293: +
batch_normalization_8_2420295: *
conv2d_5_2420298: 0
conv2d_5_2420300:0(
conv2d_5_2420302:??0+
batch_normalization_9_2420305:0+
batch_normalization_9_2420307:0+
batch_normalization_9_2420309:0+
batch_normalization_9_2420311:0*
conv2d_6_2420314:0@
conv2d_6_2420316:@&
conv2d_6_2420318:@@@,
batch_normalization_10_2420321:@,
batch_normalization_10_2420323:@,
batch_normalization_10_2420325:@,
batch_normalization_10_2420327:@*
conv2d_7_2420330:@
conv2d_7_2420332:&
conv2d_7_2420334:  ,
batch_normalization_11_2420337:,
batch_normalization_11_2420339:,
batch_normalization_11_2420341:,
batch_normalization_11_2420343:4
conv2d_transpose_5_2420346:(
conv2d_transpose_5_2420348:0
conv2d_transpose_5_2420350:@@,
batch_normalization_12_2420353:,
batch_normalization_12_2420355:,
batch_normalization_12_2420357:,
batch_normalization_12_2420359:4
conv2d_transpose_6_2420362: (
conv2d_transpose_6_2420364: 2
conv2d_transpose_6_2420366:?? ,
batch_normalization_13_2420369: ,
batch_normalization_13_2420371: ,
batch_normalization_13_2420373: ,
batch_normalization_13_2420375: 4
conv2d_transpose_7_2420378:0 (
conv2d_transpose_7_2420380:02
conv2d_transpose_7_2420382:??0,
batch_normalization_14_2420385:0,
batch_normalization_14_2420387:0,
batch_normalization_14_2420389:0,
batch_normalization_14_2420391:04
conv2d_transpose_8_2420394:@0(
conv2d_transpose_8_2420396:@2
conv2d_transpose_8_2420398:??@,
batch_normalization_15_2420401:@,
batch_normalization_15_2420403:@,
batch_normalization_15_2420405:@,
batch_normalization_15_2420407:@4
conv2d_transpose_9_2420410:@(
conv2d_transpose_9_2420412:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_2420282conv2d_4_2420284conv2d_4_2420286*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_24191002"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_8_2420289batch_normalization_8_2420291batch_normalization_8_2420293batch_normalization_8_2420295*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24201332/
-batch_normalization_8/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_5_2420298conv2d_5_2420300conv2d_5_2420302*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_24191532"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_2420305batch_normalization_9_2420307batch_normalization_9_2420309batch_normalization_9_2420311*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24200772/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_6_2420314conv2d_6_2420316conv2d_6_2420318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_24192062"
 conv2d_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_2420321batch_normalization_10_2420323batch_normalization_10_2420325batch_normalization_10_2420327*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_242002120
.batch_normalization_10/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_7_2420330conv2d_7_2420332conv2d_7_2420334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_24192592"
 conv2d_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_2420337batch_normalization_11_2420339batch_normalization_11_2420341batch_normalization_11_2420343*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_241996520
.batch_normalization_11/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_5_2420346conv2d_transpose_5_2420348conv2d_transpose_5_2420350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_24193242,
*conv2d_transpose_5/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_12_2420353batch_normalization_12_2420355batch_normalization_12_2420357batch_normalization_12_2420359*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_241991020
.batch_normalization_12/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_transpose_6_2420362conv2d_transpose_6_2420364conv2d_transpose_6_2420366*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_24193892,
*conv2d_transpose_6/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_13_2420369batch_normalization_13_2420371batch_normalization_13_2420373batch_normalization_13_2420375*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_241985520
.batch_normalization_13/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_transpose_7_2420378conv2d_transpose_7_2420380conv2d_transpose_7_2420382*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_24194542,
*conv2d_transpose_7/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_14_2420385batch_normalization_14_2420387batch_normalization_14_2420389batch_normalization_14_2420391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_241980020
.batch_normalization_14/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_transpose_8_2420394conv2d_transpose_8_2420396conv2d_transpose_8_2420398*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_24195192,
*conv2d_transpose_8/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_15_2420401batch_normalization_15_2420403batch_normalization_15_2420405batch_normalization_15_2420407*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_241974520
.batch_normalization_15/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv2d_transpose_9_2420410conv2d_transpose_9_2420412*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_24195772,
*conv2d_transpose_9/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422621

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
̆
?
 __inference__traced_save_2423915
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_6_kernel_read_readvariableop6
2savev2_conv2d_transpose_6_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_7_kernel_read_readvariableop6
2savev2_conv2d_transpose_7_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_8_kernel_read_readvariableop6
2savev2_conv2d_transpose_8_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_9_kernel_read_readvariableop6
2savev2_conv2d_transpose_9_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop7
3savev2_conv2d_4_p_re_lu_8_alpha_read_readvariableop7
3savev2_conv2d_5_p_re_lu_9_alpha_read_readvariableop8
4savev2_conv2d_6_p_re_lu_10_alpha_read_readvariableop8
4savev2_conv2d_7_p_re_lu_11_alpha_read_readvariableopB
>savev2_conv2d_transpose_5_p_re_lu_12_alpha_read_readvariableopB
>savev2_conv2d_transpose_6_p_re_lu_13_alpha_read_readvariableopB
>savev2_conv2d_transpose_7_p_re_lu_14_alpha_read_readvariableopB
>savev2_conv2d_transpose_8_p_re_lu_15_alpha_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/37/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop4savev2_conv2d_transpose_6_kernel_read_readvariableop2savev2_conv2d_transpose_6_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop4savev2_conv2d_transpose_7_kernel_read_readvariableop2savev2_conv2d_transpose_7_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop4savev2_conv2d_transpose_8_kernel_read_readvariableop2savev2_conv2d_transpose_8_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop4savev2_conv2d_transpose_9_kernel_read_readvariableop2savev2_conv2d_transpose_9_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop'savev2_sgd_momentum_read_readvariableop3savev2_conv2d_4_p_re_lu_8_alpha_read_readvariableop3savev2_conv2d_5_p_re_lu_9_alpha_read_readvariableop4savev2_conv2d_6_p_re_lu_10_alpha_read_readvariableop4savev2_conv2d_7_p_re_lu_11_alpha_read_readvariableop>savev2_conv2d_transpose_5_p_re_lu_12_alpha_read_readvariableop>savev2_conv2d_transpose_6_p_re_lu_13_alpha_read_readvariableop>savev2_conv2d_transpose_7_p_re_lu_14_alpha_read_readvariableop>savev2_conv2d_transpose_8_p_re_lu_15_alpha_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : 0:0:0:0:0:0:0@:@:@:@:@:@:@:::::::::::: : : : : : :0 :0:0:0:0:0:@0:@:@:@:@:@:@:: : : :?? :??0:@@@:  :@@:?? :??0:??@: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0: 	

_output_shapes
:0: 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
:0 : &

_output_shapes
:0: '

_output_shapes
:0: (

_output_shapes
:0: )

_output_shapes
:0: *

_output_shapes
:0:,+(
&
_output_shapes
:@0: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@: 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :*6&
$
_output_shapes
:?? :*7&
$
_output_shapes
:??0:(8$
"
_output_shapes
:@@@:(9$
"
_output_shapes
:  :(:$
"
_output_shapes
:@@:*;&
$
_output_shapes
:?? :*<&
$
_output_shapes
:??0:*=&
$
_output_shapes
:??@:>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: 
?"
?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2419519

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@:
"p_re_lu_15_readvariableop_resource:??@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_15/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddx
p_re_lu_15/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/Relu?
p_re_lu_15/ReadVariableOpReadVariableOp"p_re_lu_15_readvariableop_resource*$
_output_shapes
:??@*
dtype02
p_re_lu_15/ReadVariableOpy
p_re_lu_15/NegNeg!p_re_lu_15/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
p_re_lu_15/Negy
p_re_lu_15/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/Neg_1?
p_re_lu_15/Relu_1Relup_re_lu_15/Neg_1:y:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/Relu_1?
p_re_lu_15/mulMulp_re_lu_15/Neg:y:0p_re_lu_15/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/mul?
p_re_lu_15/addAddV2p_re_lu_15/Relu:activations:0p_re_lu_15/mul:z:0*
T0*1
_output_shapes
:???????????@2
p_re_lu_15/addw
IdentityIdentityp_re_lu_15/add:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????0: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_15/ReadVariableOpp_re_lu_15/ReadVariableOp:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_9_layer_call_fn_2422190

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24191782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_11_layer_call_fn_2422470

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_24177592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
4__inference_conv2d_transpose_5_layer_call_fn_2422592

inputs!
unknown:
	unknown_0:
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_24179552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2419855

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_2422050

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24201332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2423659

inputs/
readvariableop_resource:??@
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??@*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????@2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2423526

inputs-
readvariableop_resource:  
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:  *
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:  2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????  2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2423545

inputs-
readvariableop_resource:@@
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:@@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????@@2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
4__inference_conv2d_transpose_7_layer_call_fn_2423028

inputs!
unknown:0 
	unknown_0:0!
	unknown_1:??0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_24185172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+??????????????????????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_9_layer_call_fn_2423457

inputs!
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_24195772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_9_layer_call_fn_2422164

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24174272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_6_layer_call_fn_2422821

inputs!
unknown: 
	unknown_0: !
	unknown_1:?? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_24193892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?5
D__inference_model_1_layer_call_and_return_conditional_losses_2421355

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: B
*conv2d_4_p_re_lu_8_readvariableop_resource:?? ;
-batch_normalization_8_readvariableop_resource: =
/batch_normalization_8_readvariableop_1_resource: L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_5_conv2d_readvariableop_resource: 06
(conv2d_5_biasadd_readvariableop_resource:0B
*conv2d_5_p_re_lu_9_readvariableop_resource:??0;
-batch_normalization_9_readvariableop_resource:0=
/batch_normalization_9_readvariableop_1_resource:0L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_6_conv2d_readvariableop_resource:0@6
(conv2d_6_biasadd_readvariableop_resource:@A
+conv2d_6_p_re_lu_10_readvariableop_resource:@@@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@6
(conv2d_7_biasadd_readvariableop_resource:A
+conv2d_7_p_re_lu_11_readvariableop_resource:  <
.batch_normalization_11_readvariableop_resource:>
0batch_normalization_11_readvariableop_1_resource:M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:K
5conv2d_transpose_5_p_re_lu_12_readvariableop_resource:@@<
.batch_normalization_12_readvariableop_resource:>
0batch_normalization_12_readvariableop_1_resource:M
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_6_biasadd_readvariableop_resource: M
5conv2d_transpose_6_p_re_lu_13_readvariableop_resource:?? <
.batch_normalization_13_readvariableop_resource: >
0batch_normalization_13_readvariableop_1_resource: M
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource: U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:0 @
2conv2d_transpose_7_biasadd_readvariableop_resource:0M
5conv2d_transpose_7_p_re_lu_14_readvariableop_resource:??0<
.batch_normalization_14_readvariableop_resource:0>
0batch_normalization_14_readvariableop_1_resource:0M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:0O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:0U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource:@0@
2conv2d_transpose_8_biasadd_readvariableop_resource:@M
5conv2d_transpose_8_p_re_lu_15_readvariableop_resource:??@<
.batch_normalization_15_readvariableop_resource:@>
0batch_normalization_15_readvariableop_1_resource:@M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_9_biasadd_readvariableop_resource:
identity??6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?!conv2d_4/p_re_lu_8/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?!conv2d_5/p_re_lu_9/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?"conv2d_6/p_re_lu_10/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?"conv2d_7/p_re_lu_11/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?,conv2d_transpose_5/p_re_lu_12/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?,conv2d_transpose_6/p_re_lu_13/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?,conv2d_transpose_7/p_re_lu_14/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?,conv2d_transpose_8/p_re_lu_15/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/BiasAdd?
conv2d_4/p_re_lu_8/ReluReluconv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/Relu?
!conv2d_4/p_re_lu_8/ReadVariableOpReadVariableOp*conv2d_4_p_re_lu_8_readvariableop_resource*$
_output_shapes
:?? *
dtype02#
!conv2d_4/p_re_lu_8/ReadVariableOp?
conv2d_4/p_re_lu_8/NegNeg)conv2d_4/p_re_lu_8/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
conv2d_4/p_re_lu_8/Neg?
conv2d_4/p_re_lu_8/Neg_1Negconv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/Neg_1?
conv2d_4/p_re_lu_8/Relu_1Reluconv2d_4/p_re_lu_8/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/Relu_1?
conv2d_4/p_re_lu_8/mulMulconv2d_4/p_re_lu_8/Neg:y:0'conv2d_4/p_re_lu_8/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/mul?
conv2d_4/p_re_lu_8/addAddV2%conv2d_4/p_re_lu_8/Relu:activations:0conv2d_4/p_re_lu_8/mul:z:0*
T0*1
_output_shapes
:??????????? 2
conv2d_4/p_re_lu_8/add?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/p_re_lu_8/add:z:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_5/BiasAdd?
conv2d_5/p_re_lu_9/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/Relu?
!conv2d_5/p_re_lu_9/ReadVariableOpReadVariableOp*conv2d_5_p_re_lu_9_readvariableop_resource*$
_output_shapes
:??0*
dtype02#
!conv2d_5/p_re_lu_9/ReadVariableOp?
conv2d_5/p_re_lu_9/NegNeg)conv2d_5/p_re_lu_9/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02
conv2d_5/p_re_lu_9/Neg?
conv2d_5/p_re_lu_9/Neg_1Negconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/Neg_1?
conv2d_5/p_re_lu_9/Relu_1Reluconv2d_5/p_re_lu_9/Neg_1:y:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/Relu_1?
conv2d_5/p_re_lu_9/mulMulconv2d_5/p_re_lu_9/Neg:y:0'conv2d_5/p_re_lu_9/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/mul?
conv2d_5/p_re_lu_9/addAddV2%conv2d_5/p_re_lu_9/Relu:activations:0conv2d_5/p_re_lu_9/mul:z:0*
T0*1
_output_shapes
:???????????02
conv2d_5/p_re_lu_9/add?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/p_re_lu_9/add:z:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/BiasAdd?
conv2d_6/p_re_lu_10/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/Relu?
"conv2d_6/p_re_lu_10/ReadVariableOpReadVariableOp+conv2d_6_p_re_lu_10_readvariableop_resource*"
_output_shapes
:@@@*
dtype02$
"conv2d_6/p_re_lu_10/ReadVariableOp?
conv2d_6/p_re_lu_10/NegNeg*conv2d_6/p_re_lu_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@@2
conv2d_6/p_re_lu_10/Neg?
conv2d_6/p_re_lu_10/Neg_1Negconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/Neg_1?
conv2d_6/p_re_lu_10/Relu_1Reluconv2d_6/p_re_lu_10/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/Relu_1?
conv2d_6/p_re_lu_10/mulMulconv2d_6/p_re_lu_10/Neg:y:0(conv2d_6/p_re_lu_10/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/mul?
conv2d_6/p_re_lu_10/addAddV2&conv2d_6/p_re_lu_10/Relu:activations:0conv2d_6/p_re_lu_10/mul:z:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_6/p_re_lu_10/add?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_6/p_re_lu_10/add:z:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/BiasAdd?
conv2d_7/p_re_lu_11/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/Relu?
"conv2d_7/p_re_lu_11/ReadVariableOpReadVariableOp+conv2d_7_p_re_lu_11_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv2d_7/p_re_lu_11/ReadVariableOp?
conv2d_7/p_re_lu_11/NegNeg*conv2d_7/p_re_lu_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2
conv2d_7/p_re_lu_11/Neg?
conv2d_7/p_re_lu_11/Neg_1Negconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/Neg_1?
conv2d_7/p_re_lu_11/Relu_1Reluconv2d_7/p_re_lu_11/Neg_1:y:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/Relu_1?
conv2d_7/p_re_lu_11/mulMulconv2d_7/p_re_lu_11/Neg:y:0(conv2d_7/p_re_lu_11/Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/mul?
conv2d_7/p_re_lu_11/addAddV2&conv2d_7/p_re_lu_11/Relu:activations:0conv2d_7/p_re_lu_11/mul:z:0*
T0*/
_output_shapes
:?????????  2
conv2d_7/p_re_lu_11/add?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_7/p_re_lu_11/add:z:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3?
conv2d_transpose_5/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slicez
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_5/stack/1z
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_5/BiasAdd?
"conv2d_transpose_5/p_re_lu_12/ReluRelu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2$
"conv2d_transpose_5/p_re_lu_12/Relu?
,conv2d_transpose_5/p_re_lu_12/ReadVariableOpReadVariableOp5conv2d_transpose_5_p_re_lu_12_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv2d_transpose_5/p_re_lu_12/ReadVariableOp?
!conv2d_transpose_5/p_re_lu_12/NegNeg4conv2d_transpose_5/p_re_lu_12/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2#
!conv2d_transpose_5/p_re_lu_12/Neg?
#conv2d_transpose_5/p_re_lu_12/Neg_1Neg#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2%
#conv2d_transpose_5/p_re_lu_12/Neg_1?
$conv2d_transpose_5/p_re_lu_12/Relu_1Relu'conv2d_transpose_5/p_re_lu_12/Neg_1:y:0*
T0*/
_output_shapes
:?????????@@2&
$conv2d_transpose_5/p_re_lu_12/Relu_1?
!conv2d_transpose_5/p_re_lu_12/mulMul%conv2d_transpose_5/p_re_lu_12/Neg:y:02conv2d_transpose_5/p_re_lu_12/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2#
!conv2d_transpose_5/p_re_lu_12/mul?
!conv2d_transpose_5/p_re_lu_12/addAddV20conv2d_transpose_5/p_re_lu_12/Relu:activations:0%conv2d_transpose_5/p_re_lu_12/mul:z:0*
T0*/
_output_shapes
:?????????@@2#
!conv2d_transpose_5/p_re_lu_12/add?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_5/p_re_lu_12/add:z:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3?
conv2d_transpose_6/ShapeShape+batch_normalization_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slice{
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_6/stack/1{
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose_6/BiasAdd?
"conv2d_transpose_6/p_re_lu_13/ReluRelu#conv2d_transpose_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2$
"conv2d_transpose_6/p_re_lu_13/Relu?
,conv2d_transpose_6/p_re_lu_13/ReadVariableOpReadVariableOp5conv2d_transpose_6_p_re_lu_13_readvariableop_resource*$
_output_shapes
:?? *
dtype02.
,conv2d_transpose_6/p_re_lu_13/ReadVariableOp?
!conv2d_transpose_6/p_re_lu_13/NegNeg4conv2d_transpose_6/p_re_lu_13/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2#
!conv2d_transpose_6/p_re_lu_13/Neg?
#conv2d_transpose_6/p_re_lu_13/Neg_1Neg#conv2d_transpose_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2%
#conv2d_transpose_6/p_re_lu_13/Neg_1?
$conv2d_transpose_6/p_re_lu_13/Relu_1Relu'conv2d_transpose_6/p_re_lu_13/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2&
$conv2d_transpose_6/p_re_lu_13/Relu_1?
!conv2d_transpose_6/p_re_lu_13/mulMul%conv2d_transpose_6/p_re_lu_13/Neg:y:02conv2d_transpose_6/p_re_lu_13/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2#
!conv2d_transpose_6/p_re_lu_13/mul?
!conv2d_transpose_6/p_re_lu_13/addAddV20conv2d_transpose_6/p_re_lu_13/Relu:activations:0%conv2d_transpose_6/p_re_lu_13/mul:z:0*
T0*1
_output_shapes
:??????????? 2#
!conv2d_transpose_6/p_re_lu_13/add?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_6/p_re_lu_13/add:z:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3?
conv2d_transpose_7/ShapeShape+batch_normalization_13/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_7/Shape?
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_7/strided_slice/stack?
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_1?
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_2?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_7/strided_slice{
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_7/stack/1{
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_7/stack/2z
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_7/stack/3?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_7/stack?
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_7/strided_slice_1/stack?
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_1?
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_2?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_7/strided_slice_1?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype024
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_13/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2%
#conv2d_transpose_7/conv2d_transpose?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)conv2d_transpose_7/BiasAdd/ReadVariableOp?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_transpose_7/BiasAdd?
"conv2d_transpose_7/p_re_lu_14/ReluRelu#conv2d_transpose_7/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02$
"conv2d_transpose_7/p_re_lu_14/Relu?
,conv2d_transpose_7/p_re_lu_14/ReadVariableOpReadVariableOp5conv2d_transpose_7_p_re_lu_14_readvariableop_resource*$
_output_shapes
:??0*
dtype02.
,conv2d_transpose_7/p_re_lu_14/ReadVariableOp?
!conv2d_transpose_7/p_re_lu_14/NegNeg4conv2d_transpose_7/p_re_lu_14/ReadVariableOp:value:0*
T0*$
_output_shapes
:??02#
!conv2d_transpose_7/p_re_lu_14/Neg?
#conv2d_transpose_7/p_re_lu_14/Neg_1Neg#conv2d_transpose_7/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02%
#conv2d_transpose_7/p_re_lu_14/Neg_1?
$conv2d_transpose_7/p_re_lu_14/Relu_1Relu'conv2d_transpose_7/p_re_lu_14/Neg_1:y:0*
T0*1
_output_shapes
:???????????02&
$conv2d_transpose_7/p_re_lu_14/Relu_1?
!conv2d_transpose_7/p_re_lu_14/mulMul%conv2d_transpose_7/p_re_lu_14/Neg:y:02conv2d_transpose_7/p_re_lu_14/Relu_1:activations:0*
T0*1
_output_shapes
:???????????02#
!conv2d_transpose_7/p_re_lu_14/mul?
!conv2d_transpose_7/p_re_lu_14/addAddV20conv2d_transpose_7/p_re_lu_14/Relu:activations:0%conv2d_transpose_7/p_re_lu_14/mul:z:0*
T0*1
_output_shapes
:???????????02#
!conv2d_transpose_7/p_re_lu_14/add?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
:0*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
:0*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_7/p_re_lu_14/add:z:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3?
conv2d_transpose_8/ShapeShape+batch_normalization_14/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_8/Shape?
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_8/strided_slice/stack?
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_1?
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_2?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_8/strided_slice{
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_8/stack/1{
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_8/stack/2z
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_8/stack/3?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_8/stack?
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_8/strided_slice_1/stack?
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_1?
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_2?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_8/strided_slice_1?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype024
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2%
#conv2d_transpose_8/conv2d_transpose?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_8/BiasAdd/ReadVariableOp?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_8/BiasAdd?
"conv2d_transpose_8/p_re_lu_15/ReluRelu#conv2d_transpose_8/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2$
"conv2d_transpose_8/p_re_lu_15/Relu?
,conv2d_transpose_8/p_re_lu_15/ReadVariableOpReadVariableOp5conv2d_transpose_8_p_re_lu_15_readvariableop_resource*$
_output_shapes
:??@*
dtype02.
,conv2d_transpose_8/p_re_lu_15/ReadVariableOp?
!conv2d_transpose_8/p_re_lu_15/NegNeg4conv2d_transpose_8/p_re_lu_15/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2#
!conv2d_transpose_8/p_re_lu_15/Neg?
#conv2d_transpose_8/p_re_lu_15/Neg_1Neg#conv2d_transpose_8/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#conv2d_transpose_8/p_re_lu_15/Neg_1?
$conv2d_transpose_8/p_re_lu_15/Relu_1Relu'conv2d_transpose_8/p_re_lu_15/Neg_1:y:0*
T0*1
_output_shapes
:???????????@2&
$conv2d_transpose_8/p_re_lu_15/Relu_1?
!conv2d_transpose_8/p_re_lu_15/mulMul%conv2d_transpose_8/p_re_lu_15/Neg:y:02conv2d_transpose_8/p_re_lu_15/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2#
!conv2d_transpose_8/p_re_lu_15/mul?
!conv2d_transpose_8/p_re_lu_15/addAddV20conv2d_transpose_8/p_re_lu_15/Relu:activations:0%conv2d_transpose_8/p_re_lu_15/mul:z:0*
T0*1
_output_shapes
:???????????@2#
!conv2d_transpose_8/p_re_lu_15/add?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3%conv2d_transpose_8/p_re_lu_15/add:z:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
conv2d_transpose_9/ShapeShape+batch_normalization_15/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slice{
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/1{
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/2z
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_15/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_9/BiasAdd?
conv2d_transpose_9/SigmoidSigmoid#conv2d_transpose_9/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_9/Sigmoid?
IdentityIdentityconv2d_transpose_9/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp"^conv2d_4/p_re_lu_8/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp"^conv2d_5/p_re_lu_9/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp#^conv2d_6/p_re_lu_10/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp#^conv2d_7/p_re_lu_11/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp-^conv2d_transpose_5/p_re_lu_12/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp-^conv2d_transpose_6/p_re_lu_13/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp-^conv2d_transpose_7/p_re_lu_14/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp-^conv2d_transpose_8/p_re_lu_15/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2F
!conv2d_4/p_re_lu_8/ReadVariableOp!conv2d_4/p_re_lu_8/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2F
!conv2d_5/p_re_lu_9/ReadVariableOp!conv2d_5/p_re_lu_9/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2H
"conv2d_6/p_re_lu_10/ReadVariableOp"conv2d_6/p_re_lu_10/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2H
"conv2d_7/p_re_lu_11/ReadVariableOp"conv2d_7/p_re_lu_11/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_5/p_re_lu_12/ReadVariableOp,conv2d_transpose_5/p_re_lu_12/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_6/p_re_lu_13/ReadVariableOp,conv2d_transpose_6/p_re_lu_13/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_7/p_re_lu_14/ReadVariableOp,conv2d_transpose_7/p_re_lu_14/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2\
,conv2d_transpose_8/p_re_lu_15/ReadVariableOp,conv2d_transpose_8/p_re_lu_15/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2419259

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:8
"p_re_lu_11_readvariableop_resource:  
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_11/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAddv
p_re_lu_11/ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/Relu?
p_re_lu_11/ReadVariableOpReadVariableOp"p_re_lu_11_readvariableop_resource*"
_output_shapes
:  *
dtype02
p_re_lu_11/ReadVariableOpw
p_re_lu_11/NegNeg!p_re_lu_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2
p_re_lu_11/Negw
p_re_lu_11/Neg_1NegBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/Neg_1~
p_re_lu_11/Relu_1Relup_re_lu_11/Neg_1:y:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/Relu_1?
p_re_lu_11/mulMulp_re_lu_11/Neg:y:0p_re_lu_11/Relu_1:activations:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/mul?
p_re_lu_11/addAddV2p_re_lu_11/Relu:activations:0p_re_lu_11/mul:z:0*
T0*/
_output_shapes
:?????????  2
p_re_lu_11/addu
IdentityIdentityp_re_lu_11/add:z:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@@@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp26
p_re_lu_11/ReadVariableOpp_re_lu_11/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2419577

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoidp
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_7_layer_call_fn_2423039

inputs!
unknown:0 
	unknown_0:0!
	unknown_1:??0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_24194542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2417879

inputs-
readvariableop_resource:@@
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:@@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????@@2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2417803

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_10_layer_call_fn_2422330

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_24176372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421980

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2417427

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422151

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2418365

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2417471

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2420021

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_2421915

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 9
!p_re_lu_8_readvariableop_resource:?? 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?p_re_lu_8/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddv
p_re_lu_8/ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/Relu?
p_re_lu_8/ReadVariableOpReadVariableOp!p_re_lu_8_readvariableop_resource*$
_output_shapes
:?? *
dtype02
p_re_lu_8/ReadVariableOpv
p_re_lu_8/NegNeg p_re_lu_8/ReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
p_re_lu_8/Negw
p_re_lu_8/Neg_1NegBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/Neg_1}
p_re_lu_8/Relu_1Relup_re_lu_8/Neg_1:y:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/Relu_1?
p_re_lu_8/mulMulp_re_lu_8/Neg:y:0p_re_lu_8/Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/mul?
p_re_lu_8/addAddV2p_re_lu_8/Relu:activations:0p_re_lu_8/mul:z:0*
T0*1
_output_shapes
:??????????? 2
p_re_lu_8/addv
IdentityIdentityp_re_lu_8/add:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^p_re_lu_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp24
p_re_lu_8/ReadVariableOpp_re_lu_8/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?)
?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2418517

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0*
p_re_lu_14_2418513:??0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?"p_re_lu_14/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :02	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????02	
BiasAdd?
"p_re_lu_14/StatefulPartitionedCallStatefulPartitionedCallBiasAdd:output:0p_re_lu_14_2418513*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_24185122$
"p_re_lu_14/StatefulPartitionedCall?
IdentityIdentity+p_re_lu_14/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp#^p_re_lu_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+??????????????????????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2H
"p_re_lu_14/StatefulPartitionedCall"p_re_lu_14/StatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_12_layer_call_fn_2422727

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_24199102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?&
?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2423415

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?.
?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2422550

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:8
"p_re_lu_12_readvariableop_resource:@@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?p_re_lu_12/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
p_re_lu_12/ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
p_re_lu_12/Relu?
p_re_lu_12/ReadVariableOpReadVariableOp"p_re_lu_12_readvariableop_resource*"
_output_shapes
:@@*
dtype02
p_re_lu_12/ReadVariableOpw
p_re_lu_12/NegNeg!p_re_lu_12/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
p_re_lu_12/Neg?
p_re_lu_12/Neg_1NegBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
p_re_lu_12/Neg_1?
p_re_lu_12/Relu_1Relup_re_lu_12/Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2
p_re_lu_12/Relu_1?
p_re_lu_12/mulMulp_re_lu_12/Neg:y:0p_re_lu_12/Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/mul?
p_re_lu_12/addAddV2p_re_lu_12/Relu:activations:0p_re_lu_12/mul:z:0*
T0*/
_output_shapes
:?????????@@2
p_re_lu_12/addu
IdentityIdentityp_re_lu_12/add:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp^p_re_lu_12/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp26
p_re_lu_12/ReadVariableOpp_re_lu_12/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_11_layer_call_fn_2422509

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_24199652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_9_layer_call_fn_2422203

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_24200772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2423469

inputs/
readvariableop_resource:?? 
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:?? *
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:?? 2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:??????????? 2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:??????????? 2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4????????????????????????????????????: 2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_10_layer_call_fn_2422343

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_24192312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
,__inference_p_re_lu_12_layer_call_fn_2423571

inputs
unknown:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_24179502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2423557

inputs-
readvariableop_resource:@@
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:@@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:@@2
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+???????????????????????????2
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????@@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:?????????@@2
addj
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2418793

inputs/
readvariableop_resource:??@
identity??ReadVariableOph
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu~
ReadVariableOpReadVariableOpreadvariableop_resource*$
_output_shapes
:??@*
dtype02
ReadVariableOpX
NegNegReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
Negi
Neg_1Neginputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Neg_1o
Relu_1Relu	Neg_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu_1l
mulMulNeg:y:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mull
addAddV2Relu:activations:0mul:z:0*
T0*1
_output_shapes
:???????????@2
addl
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_2:
serving_default_input_2:0???????????P
conv2d_transpose_9:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
layer_with_weights-16
layer-17
	optimizer
loss
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

activation

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
*
activation

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
:
activation

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
J
activation

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Z
activation

[kernel
\bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
j
activation

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
z
activation

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?
activation
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
9
	?iter

?decay
?momentum"
	optimizer
 "
trackable_list_wrapper
?
0
1
?2
"3
#4
+5
,6
?7
28
39
;10
<11
?12
B13
C14
K15
L16
?17
R18
S19
[20
\21
?22
b23
c24
k25
l26
?27
r28
s29
{30
|31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41"
trackable_list_wrapper
?
0
1
?2
"3
#4
$5
%6
+7
,8
?9
210
311
412
513
;14
<15
?16
B17
C18
D19
E20
K21
L22
?23
R24
S25
T26
U27
[28
\29
?30
b31
c32
d33
e34
k35
l36
?37
r38
s39
t40
u41
{42
|43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?metrics
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
):' 2conv2d_4/kernel
: 2conv2d_4/bias
6
0
1
?2"
trackable_list_wrapper
6
0
1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
trainable_variables
 ?layer_regularization_losses
	variables
?non_trainable_variables
regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
.
"0
#1"
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
&trainable_variables
 ?layer_regularization_losses
'	variables
?non_trainable_variables
(regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
):' 02conv2d_5/kernel
:02conv2d_5/bias
6
+0
,1
?2"
trackable_list_wrapper
6
+0
,1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
-trainable_variables
 ?layer_regularization_losses
.	variables
?non_trainable_variables
/regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_9/gamma
(:&02batch_normalization_9/beta
1:/0 (2!batch_normalization_9/moving_mean
5:30 (2%batch_normalization_9/moving_variance
.
20
31"
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
6trainable_variables
 ?layer_regularization_losses
7	variables
?non_trainable_variables
8regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
):'0@2conv2d_6/kernel
:@2conv2d_6/bias
6
;0
<1
?2"
trackable_list_wrapper
6
;0
<1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
=trainable_variables
 ?layer_regularization_losses
>	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Ftrainable_variables
 ?layer_regularization_losses
G	variables
?non_trainable_variables
Hregularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
):'@2conv2d_7/kernel
:2conv2d_7/bias
6
K0
L1
?2"
trackable_list_wrapper
6
K0
L1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Mtrainable_variables
 ?layer_regularization_losses
N	variables
?non_trainable_variables
Oregularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
.
R0
S1"
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Vtrainable_variables
 ?layer_regularization_losses
W	variables
?non_trainable_variables
Xregularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
3:12conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
6
[0
\1
?2"
trackable_list_wrapper
6
[0
\1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
]trainable_variables
 ?layer_regularization_losses
^	variables
?non_trainable_variables
_regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
.
b0
c1"
trackable_list_wrapper
<
b0
c1
d2
e3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
ftrainable_variables
 ?layer_regularization_losses
g	variables
?non_trainable_variables
hregularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
3:1 2conv2d_transpose_6/kernel
%:# 2conv2d_transpose_6/bias
6
k0
l1
?2"
trackable_list_wrapper
6
k0
l1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
mtrainable_variables
 ?layer_regularization_losses
n	variables
?non_trainable_variables
oregularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_13/gamma
):' 2batch_normalization_13/beta
2:0  (2"batch_normalization_13/moving_mean
6:4  (2&batch_normalization_13/moving_variance
.
r0
s1"
trackable_list_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
vtrainable_variables
 ?layer_regularization_losses
w	variables
?non_trainable_variables
xregularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
3:10 2conv2d_transpose_7/kernel
%:#02conv2d_transpose_7/bias
6
{0
|1
?2"
trackable_list_wrapper
6
{0
|1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
}trainable_variables
 ?layer_regularization_losses
~	variables
?non_trainable_variables
regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(02batch_normalization_14/gamma
):'02batch_normalization_14/beta
2:00 (2"batch_normalization_14/moving_mean
6:40 (2&batch_normalization_14/moving_variance
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?alpha
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
3:1@02conv2d_transpose_8/kernel
%:#@2conv2d_transpose_8/bias
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_15/gamma
):'@2batch_normalization_15/beta
2:0@ (2"batch_normalization_15/moving_mean
6:4@ (2&batch_normalization_15/moving_variance
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1@2conv2d_transpose_9/kernel
%:#2conv2d_transpose_9/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/momentum
0:.?? 2conv2d_4/p_re_lu_8/alpha
0:.??02conv2d_5/p_re_lu_9/alpha
/:-@@@2conv2d_6/p_re_lu_10/alpha
/:-  2conv2d_7/p_re_lu_11/alpha
9:7@@2#conv2d_transpose_5/p_re_lu_12/alpha
;:9?? 2#conv2d_transpose_6/p_re_lu_13/alpha
;:9??02#conv2d_transpose_7/p_re_lu_14/alpha
;:9??@2#conv2d_transpose_8/p_re_lu_15/alpha
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
$0
%1
42
53
D4
E5
T6
U7
d8
e9
t10
u11
?12
?13
?14
?15"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Z0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
j0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
z0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?B?
"__inference__wrapped_model_2417199input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_model_1_layer_call_and_return_conditional_losses_2421355
D__inference_model_1_layer_call_and_return_conditional_losses_2421655
D__inference_model_1_layer_call_and_return_conditional_losses_2420793
D__inference_model_1_layer_call_and_return_conditional_losses_2420930?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_model_1_layer_call_fn_2419703
)__inference_model_1_layer_call_fn_2421776
)__inference_model_1_layer_call_fn_2421897
)__inference_model_1_layer_call_fn_2420656?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_2421915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_4_layer_call_fn_2421926?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421944
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421962
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421980
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421998?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_8_layer_call_fn_2422011
7__inference_batch_normalization_8_layer_call_fn_2422024
7__inference_batch_normalization_8_layer_call_fn_2422037
7__inference_batch_normalization_8_layer_call_fn_2422050?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2422068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_5_layer_call_fn_2422079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422097
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422115
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422133
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422151?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_9_layer_call_fn_2422164
7__inference_batch_normalization_9_layer_call_fn_2422177
7__inference_batch_normalization_9_layer_call_fn_2422190
7__inference_batch_normalization_9_layer_call_fn_2422203?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2422221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_6_layer_call_fn_2422232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422250
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422268
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422286
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422304?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_10_layer_call_fn_2422317
8__inference_batch_normalization_10_layer_call_fn_2422330
8__inference_batch_normalization_10_layer_call_fn_2422343
8__inference_batch_normalization_10_layer_call_fn_2422356?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2422374?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_7_layer_call_fn_2422385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422403
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422421
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422439
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422457?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_11_layer_call_fn_2422470
8__inference_batch_normalization_11_layer_call_fn_2422483
8__inference_batch_normalization_11_layer_call_fn_2422496
8__inference_batch_normalization_11_layer_call_fn_2422509?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2422550
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2422581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_5_layer_call_fn_2422592
4__inference_conv2d_transpose_5_layer_call_fn_2422603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422621
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422639
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422657
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422675?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_12_layer_call_fn_2422688
8__inference_batch_normalization_12_layer_call_fn_2422701
8__inference_batch_normalization_12_layer_call_fn_2422714
8__inference_batch_normalization_12_layer_call_fn_2422727?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2422768
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2422799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_6_layer_call_fn_2422810
4__inference_conv2d_transpose_6_layer_call_fn_2422821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422839
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422857
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422875
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422893?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_13_layer_call_fn_2422906
8__inference_batch_normalization_13_layer_call_fn_2422919
8__inference_batch_normalization_13_layer_call_fn_2422932
8__inference_batch_normalization_13_layer_call_fn_2422945?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2422986
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2423017?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_7_layer_call_fn_2423028
4__inference_conv2d_transpose_7_layer_call_fn_2423039?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423057
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423075
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423093
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423111?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_14_layer_call_fn_2423124
8__inference_batch_normalization_14_layer_call_fn_2423137
8__inference_batch_normalization_14_layer_call_fn_2423150
8__inference_batch_normalization_14_layer_call_fn_2423163?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2423204
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2423235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_8_layer_call_fn_2423246
4__inference_conv2d_transpose_8_layer_call_fn_2423257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423275
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423293
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423311
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423329?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_15_layer_call_fn_2423342
8__inference_batch_normalization_15_layer_call_fn_2423355
8__inference_batch_normalization_15_layer_call_fn_2423368
8__inference_batch_normalization_15_layer_call_fn_2423381?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2423415
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2423439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_9_layer_call_fn_2423448
4__inference_conv2d_transpose_9_layer_call_fn_2423457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2421055input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2423469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_p_re_lu_8_layer_call_fn_2423476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2423488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_p_re_lu_9_layer_call_fn_2423495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2423507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_10_layer_call_fn_2423514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2423526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_11_layer_call_fn_2423533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2423545
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2423557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_12_layer_call_fn_2423564
,__inference_p_re_lu_12_layer_call_fn_2423571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2423583
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2423595?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_13_layer_call_fn_2423602
,__inference_p_re_lu_13_layer_call_fn_2423609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2423621
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2423633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_14_layer_call_fn_2423640
,__inference_p_re_lu_14_layer_call_fn_2423647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2423659
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2423671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_15_layer_call_fn_2423678
,__inference_p_re_lu_15_layer_call_fn_2423685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2417199?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????:?7
0?-
+?(
input_2???????????
? "Q?N
L
conv2d_transpose_96?3
conv2d_transpose_9????????????
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422250?BCDEM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422268?BCDEM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422286rBCDE;?8
1?.
(?%
inputs?????????@@@
p 
? "-?*
#? 
0?????????@@@
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2422304rBCDE;?8
1?.
(?%
inputs?????????@@@
p
? "-?*
#? 
0?????????@@@
? ?
8__inference_batch_normalization_10_layer_call_fn_2422317?BCDEM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_10_layer_call_fn_2422330?BCDEM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_10_layer_call_fn_2422343eBCDE;?8
1?.
(?%
inputs?????????@@@
p 
? " ??????????@@@?
8__inference_batch_normalization_10_layer_call_fn_2422356eBCDE;?8
1?.
(?%
inputs?????????@@@
p
? " ??????????@@@?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422403?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422421?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422439rRSTU;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2422457rRSTU;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
8__inference_batch_normalization_11_layer_call_fn_2422470?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_11_layer_call_fn_2422483?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_11_layer_call_fn_2422496eRSTU;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_11_layer_call_fn_2422509eRSTU;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422621?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422639?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422657rbcde;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2422675rbcde;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
8__inference_batch_normalization_12_layer_call_fn_2422688?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_12_layer_call_fn_2422701?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_12_layer_call_fn_2422714ebcde;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
8__inference_batch_normalization_12_layer_call_fn_2422727ebcde;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422839?rstuM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422857?rstuM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422875vrstu=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2422893vrstu=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
8__inference_batch_normalization_13_layer_call_fn_2422906?rstuM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_13_layer_call_fn_2422919?rstuM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_13_layer_call_fn_2422932irstu=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
8__inference_batch_normalization_13_layer_call_fn_2422945irstu=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423057?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423075?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423093z????=?:
3?0
*?'
inputs???????????0
p 
? "/?,
%?"
0???????????0
? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2423111z????=?:
3?0
*?'
inputs???????????0
p
? "/?,
%?"
0???????????0
? ?
8__inference_batch_normalization_14_layer_call_fn_2423124?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
8__inference_batch_normalization_14_layer_call_fn_2423137?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
8__inference_batch_normalization_14_layer_call_fn_2423150m????=?:
3?0
*?'
inputs???????????0
p 
? ""????????????0?
8__inference_batch_normalization_14_layer_call_fn_2423163m????=?:
3?0
*?'
inputs???????????0
p
? ""????????????0?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423275?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423293?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423311z????=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2423329z????=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
8__inference_batch_normalization_15_layer_call_fn_2423342?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_15_layer_call_fn_2423355?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_15_layer_call_fn_2423368m????=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
8__inference_batch_normalization_15_layer_call_fn_2423381m????=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421944?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421962?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421980v"#$%=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2421998v"#$%=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
7__inference_batch_normalization_8_layer_call_fn_2422011?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_8_layer_call_fn_2422024?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_8_layer_call_fn_2422037i"#$%=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
7__inference_batch_normalization_8_layer_call_fn_2422050i"#$%=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422097?2345M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422115?2345M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422133v2345=?:
3?0
*?'
inputs???????????0
p 
? "/?,
%?"
0???????????0
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2422151v2345=?:
3?0
*?'
inputs???????????0
p
? "/?,
%?"
0???????????0
? ?
7__inference_batch_normalization_9_layer_call_fn_2422164?2345M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
7__inference_batch_normalization_9_layer_call_fn_2422177?2345M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
7__inference_batch_normalization_9_layer_call_fn_2422190i2345=?:
3?0
*?'
inputs???????????0
p 
? ""????????????0?
7__inference_batch_normalization_9_layer_call_fn_2422203i2345=?:
3?0
*?'
inputs???????????0
p
? ""????????????0?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_2421915r?9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
*__inference_conv2d_4_layer_call_fn_2421926e?9?6
/?,
*?'
inputs???????????
? ""???????????? ?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_2422068r+,?9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????0
? ?
*__inference_conv2d_5_layer_call_fn_2422079e+,?9?6
/?,
*?'
inputs??????????? 
? ""????????????0?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_2422221p;<?9?6
/?,
*?'
inputs???????????0
? "-?*
#? 
0?????????@@@
? ?
*__inference_conv2d_6_layer_call_fn_2422232c;<?9?6
/?,
*?'
inputs???????????0
? " ??????????@@@?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_2422374nKL?7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_7_layer_call_fn_2422385aKL?7?4
-?*
(?%
inputs?????????@@@
? " ??????????  ?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2422550?[\?I?F
??<
:?7
inputs+???????????????????????????
? "-?*
#? 
0?????????@@
? ?
O__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_2422581n[\?7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????@@
? ?
4__inference_conv2d_transpose_5_layer_call_fn_2422592s[\?I?F
??<
:?7
inputs+???????????????????????????
? " ??????????@@?
4__inference_conv2d_transpose_5_layer_call_fn_2422603a[\?7?4
-?*
(?%
inputs?????????  
? " ??????????@@?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2422768?kl?I?F
??<
:?7
inputs+???????????????????????????
? "/?,
%?"
0??????????? 
? ?
O__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_2422799pkl?7?4
-?*
(?%
inputs?????????@@
? "/?,
%?"
0??????????? 
? ?
4__inference_conv2d_transpose_6_layer_call_fn_2422810ukl?I?F
??<
:?7
inputs+???????????????????????????
? ""???????????? ?
4__inference_conv2d_transpose_6_layer_call_fn_2422821ckl?7?4
-?*
(?%
inputs?????????@@
? ""???????????? ?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2422986?{|?I?F
??<
:?7
inputs+??????????????????????????? 
? "/?,
%?"
0???????????0
? ?
O__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_2423017r{|?9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????0
? ?
4__inference_conv2d_transpose_7_layer_call_fn_2423028u{|?I?F
??<
:?7
inputs+??????????????????????????? 
? ""????????????0?
4__inference_conv2d_transpose_7_layer_call_fn_2423039e{|?9?6
/?,
*?'
inputs??????????? 
? ""????????????0?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2423204????I?F
??<
:?7
inputs+???????????????????????????0
? "/?,
%?"
0???????????@
? ?
O__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_2423235t???9?6
/?,
*?'
inputs???????????0
? "/?,
%?"
0???????????@
? ?
4__inference_conv2d_transpose_8_layer_call_fn_2423246w???I?F
??<
:?7
inputs+???????????????????????????0
? ""????????????@?
4__inference_conv2d_transpose_8_layer_call_fn_2423257g???9?6
/?,
*?'
inputs???????????0
? ""????????????@?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2423415???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
O__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_2423439r??9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????
? ?
4__inference_conv2d_transpose_9_layer_call_fn_2423448???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
4__inference_conv2d_transpose_9_layer_call_fn_2423457e??9?6
/?,
*?'
inputs???????????@
? ""?????????????
D__inference_model_1_layer_call_and_return_conditional_losses_2420793?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????B??
8?5
+?(
input_2???????????
p 

 
? "/?,
%?"
0???????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_2420930?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????B??
8?5
+?(
input_2???????????
p

 
? "/?,
%?"
0???????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_2421355?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_2421655?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
)__inference_model_1_layer_call_fn_2419703?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????B??
8?5
+?(
input_2???????????
p 

 
? ""?????????????
)__inference_model_1_layer_call_fn_2420656?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????B??
8?5
+?(
input_2???????????
p

 
? ""?????????????
)__inference_model_1_layer_call_fn_2421776?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
)__inference_model_1_layer_call_fn_2421897?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
G__inference_p_re_lu_10_layer_call_and_return_conditional_losses_2423507??R?O
H?E
C?@
inputs4????????????????????????????????????
? "-?*
#? 
0?????????@@@
? ?
,__inference_p_re_lu_10_layer_call_fn_2423514z?R?O
H?E
C?@
inputs4????????????????????????????????????
? " ??????????@@@?
G__inference_p_re_lu_11_layer_call_and_return_conditional_losses_2423526??R?O
H?E
C?@
inputs4????????????????????????????????????
? "-?*
#? 
0?????????  
? ?
,__inference_p_re_lu_11_layer_call_fn_2423533z?R?O
H?E
C?@
inputs4????????????????????????????????????
? " ??????????  ?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2423545??R?O
H?E
C?@
inputs4????????????????????????????????????
? "-?*
#? 
0?????????@@
? ?
G__inference_p_re_lu_12_layer_call_and_return_conditional_losses_2423557~?I?F
??<
:?7
inputs+???????????????????????????
? "-?*
#? 
0?????????@@
? ?
,__inference_p_re_lu_12_layer_call_fn_2423564z?R?O
H?E
C?@
inputs4????????????????????????????????????
? " ??????????@@?
,__inference_p_re_lu_12_layer_call_fn_2423571q?I?F
??<
:?7
inputs+???????????????????????????
? " ??????????@@?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2423583??R?O
H?E
C?@
inputs4????????????????????????????????????
? "/?,
%?"
0??????????? 
? ?
G__inference_p_re_lu_13_layer_call_and_return_conditional_losses_2423595??I?F
??<
:?7
inputs+??????????????????????????? 
? "/?,
%?"
0??????????? 
? ?
,__inference_p_re_lu_13_layer_call_fn_2423602|?R?O
H?E
C?@
inputs4????????????????????????????????????
? ""???????????? ?
,__inference_p_re_lu_13_layer_call_fn_2423609s?I?F
??<
:?7
inputs+??????????????????????????? 
? ""???????????? ?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2423621??R?O
H?E
C?@
inputs4????????????????????????????????????
? "/?,
%?"
0???????????0
? ?
G__inference_p_re_lu_14_layer_call_and_return_conditional_losses_2423633??I?F
??<
:?7
inputs+???????????????????????????0
? "/?,
%?"
0???????????0
? ?
,__inference_p_re_lu_14_layer_call_fn_2423640|?R?O
H?E
C?@
inputs4????????????????????????????????????
? ""????????????0?
,__inference_p_re_lu_14_layer_call_fn_2423647s?I?F
??<
:?7
inputs+???????????????????????????0
? ""????????????0?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2423659??R?O
H?E
C?@
inputs4????????????????????????????????????
? "/?,
%?"
0???????????@
? ?
G__inference_p_re_lu_15_layer_call_and_return_conditional_losses_2423671??I?F
??<
:?7
inputs+???????????????????????????@
? "/?,
%?"
0???????????@
? ?
,__inference_p_re_lu_15_layer_call_fn_2423678|?R?O
H?E
C?@
inputs4????????????????????????????????????
? ""????????????@?
,__inference_p_re_lu_15_layer_call_fn_2423685s?I?F
??<
:?7
inputs+???????????????????????????@
? ""????????????@?
F__inference_p_re_lu_8_layer_call_and_return_conditional_losses_2423469??R?O
H?E
C?@
inputs4????????????????????????????????????
? "/?,
%?"
0??????????? 
? ?
+__inference_p_re_lu_8_layer_call_fn_2423476|?R?O
H?E
C?@
inputs4????????????????????????????????????
? ""???????????? ?
F__inference_p_re_lu_9_layer_call_and_return_conditional_losses_2423488??R?O
H?E
C?@
inputs4????????????????????????????????????
? "/?,
%?"
0???????????0
? ?
+__inference_p_re_lu_9_layer_call_fn_2423495|?R?O
H?E
C?@
inputs4????????????????????????????????????
? ""????????????0?
%__inference_signature_wrapper_2421055?N?"#$%+,?2345;<?BCDEKL?RSTU[\?bcdekl?rstu{|??????????????E?B
? 
;?8
6
input_2+?(
input_2???????????"Q?N
L
conv2d_transpose_96?3
conv2d_transpose_9???????????