??)
??
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??%
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
?
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_24/gamma
?
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_24/beta
?
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_24/moving_mean
?
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_24/moving_variance
?
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
: 0*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_namebatch_normalization_25/gamma
?
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_25/beta
?
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
:0*
dtype0
?
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"batch_normalization_25/moving_mean
?
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
:0*
dtype0
?
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&batch_normalization_25/moving_variance
?
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:0@*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_26/gamma
?
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_26/beta
?
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_26/moving_mean
?
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_26/moving_variance
?
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:*
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_15/kernel
?
.conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_15/bias
?
,conv2d_transpose_15/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/bias*
_output_shapes
:*
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_transpose_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_16/kernel
?
.conv2d_transpose_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_16/bias
?
,conv2d_transpose_16/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/bias*
_output_shapes
: *
dtype0
?
batch_normalization_29/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_29/gamma
?
0batch_normalization_29/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_29/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_29/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_29/beta
?
/batch_normalization_29/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_29/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_29/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_29/moving_mean
?
6batch_normalization_29/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_29/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_29/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_29/moving_variance
?
:batch_normalization_29/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_29/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0 *+
shared_nameconv2d_transpose_17/kernel
?
.conv2d_transpose_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/kernel*&
_output_shapes
:0 *
dtype0
?
conv2d_transpose_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_nameconv2d_transpose_17/bias
?
,conv2d_transpose_17/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_30/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_namebatch_normalization_30/gamma
?
0batch_normalization_30/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_30/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_30/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_30/beta
?
/batch_normalization_30/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_30/beta*
_output_shapes
:0*
dtype0
?
"batch_normalization_30/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"batch_normalization_30/moving_mean
?
6batch_normalization_30/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_30/moving_mean*
_output_shapes
:0*
dtype0
?
&batch_normalization_30/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&batch_normalization_30/moving_variance
?
:batch_normalization_30/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_30/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_transpose_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@0*+
shared_nameconv2d_transpose_18/kernel
?
.conv2d_transpose_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/kernel*&
_output_shapes
:@0*
dtype0
?
conv2d_transpose_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_18/bias
?
,conv2d_transpose_18/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_31/gamma
?
0batch_normalization_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_31/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_31/beta
?
/batch_normalization_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_31/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_31/moving_mean
?
6batch_normalization_31/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_31/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_31/moving_variance
?
:batch_normalization_31/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_31/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_19/kernel
?
.conv2d_transpose_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/kernel*&
_output_shapes
:@*
dtype0
?
conv2d_transpose_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_19/bias
?
,conv2d_transpose_19/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/bias*
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
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0

NoOpNoOp
?x
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?w
value?wB?w B?w
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5	variables
6trainable_variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
?
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
h

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
?
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
&
	?iter

?decay
?momentum
 
 
?
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
812
913
?14
@15
A16
B17
G18
H19
N20
O21
P22
Q23
V24
W25
]26
^27
_28
`29
e30
f31
l32
m33
n34
o35
t36
u37
{38
|39
}40
~41
?42
?43
?44
?45
?46
?47
?48
?49
?
0
1
!2
"3
)4
*5
06
17
88
99
?10
@11
G12
H13
N14
O15
V16
W17
]18
^19
e20
f21
l22
m23
t24
u25
{26
|27
?28
?29
?30
?31
?32
?33
?
?layer_metrics
 ?layer_regularization_losses
regularization_losses
?layers
	variables
trainable_variables
?non_trainable_variables
?metrics
 
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
	variables
trainable_variables
?non_trainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_24/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_24/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_24/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_24/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1
#2
$3

!0
"1
?
%regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
&	variables
'trainable_variables
?non_trainable_variables
?metrics
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
+regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
,	variables
-trainable_variables
?non_trainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_25/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_25/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_25/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_25/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

00
11
22
33

00
11
?
4regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
5	variables
6trainable_variables
?non_trainable_variables
?metrics
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
?
:regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
;	variables
<trainable_variables
?non_trainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_26/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_26/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_26/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_26/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1
A2
B3

?0
@1
?
Cregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
D	variables
Etrainable_variables
?non_trainable_variables
?metrics
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
?
Iregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
J	variables
Ktrainable_variables
?non_trainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_27/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_27/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_27/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_27/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1
P2
Q3

N0
O1
?
Rregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
S	variables
Ttrainable_variables
?non_trainable_variables
?metrics
fd
VARIABLE_VALUEconv2d_transpose_15/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_15/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
?
Xregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
Y	variables
Ztrainable_variables
?non_trainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_28/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_28/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_28/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_28/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1
_2
`3

]0
^1
?
aregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
b	variables
ctrainable_variables
?non_trainable_variables
?metrics
ge
VARIABLE_VALUEconv2d_transpose_16/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_16/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
?
gregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
h	variables
itrainable_variables
?non_trainable_variables
?metrics
 
hf
VARIABLE_VALUEbatch_normalization_29/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_29/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_29/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_29/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1
n2
o3

l0
m1
?
pregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
q	variables
rtrainable_variables
?non_trainable_variables
?metrics
ge
VARIABLE_VALUEconv2d_transpose_17/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_17/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
?
vregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
w	variables
xtrainable_variables
?non_trainable_variables
?metrics
 
hf
VARIABLE_VALUEbatch_normalization_30/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_30/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_30/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_30/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1
}2
~3

{0
|1
?
regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
ge
VARIABLE_VALUEconv2d_transpose_18/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_18/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
 
hf
VARIABLE_VALUEbatch_normalization_31/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_31/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_31/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_31/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?0
?1
?
?regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
ge
VARIABLE_VALUEconv2d_transpose_19/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_19/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
 
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
x
#0
$1
22
33
A4
B5
P6
Q7
_8
`9
n10
o11
}12
~13
?14
?15
(
?0
?1
?2
?3
?4
 
 
 
 
 
 
 
 

#0
$1
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 

n0
o1
 
 
 
 
 
 
 
 
 

}0
~1
 
 
 
 
 
 
 
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
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
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
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
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_input_4Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_12/kernelconv2d_12/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_transpose_15/kernelconv2d_transpose_15/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_16/kernelconv2d_transpose_16/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv2d_transpose_17/kernelconv2d_transpose_17/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv2d_transpose_18/kernelconv2d_transpose_18/biasbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_varianceconv2d_transpose_19/kernelconv2d_transpose_19/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_5324463
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp.conv2d_transpose_15/kernel/Read/ReadVariableOp,conv2d_transpose_15/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp.conv2d_transpose_16/kernel/Read/ReadVariableOp,conv2d_transpose_16/bias/Read/ReadVariableOp0batch_normalization_29/gamma/Read/ReadVariableOp/batch_normalization_29/beta/Read/ReadVariableOp6batch_normalization_29/moving_mean/Read/ReadVariableOp:batch_normalization_29/moving_variance/Read/ReadVariableOp.conv2d_transpose_17/kernel/Read/ReadVariableOp,conv2d_transpose_17/bias/Read/ReadVariableOp0batch_normalization_30/gamma/Read/ReadVariableOp/batch_normalization_30/beta/Read/ReadVariableOp6batch_normalization_30/moving_mean/Read/ReadVariableOp:batch_normalization_30/moving_variance/Read/ReadVariableOp.conv2d_transpose_18/kernel/Read/ReadVariableOp,conv2d_transpose_18/bias/Read/ReadVariableOp0batch_normalization_31/gamma/Read/ReadVariableOp/batch_normalization_31/beta/Read/ReadVariableOp6batch_normalization_31/moving_mean/Read/ReadVariableOp:batch_normalization_31/moving_variance/Read/ReadVariableOp.conv2d_transpose_19/kernel/Read/ReadVariableOp,conv2d_transpose_19/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpConst*L
TinE
C2A	*
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
 __inference__traced_save_5326825
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_12/kernelconv2d_12/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_transpose_15/kernelconv2d_transpose_15/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_16/kernelconv2d_transpose_16/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv2d_transpose_17/kernelconv2d_transpose_17/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv2d_transpose_18/kernelconv2d_transpose_18/biasbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_varianceconv2d_transpose_19/kernelconv2d_transpose_19/biasSGD/iter	SGD/decaySGD/momentumtotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4*K
TinD
B2@*
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
#__inference__traced_restore_5327024??#
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325701

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
?
?
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5322961

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
T0*/
_output_shapes
:?????????@@ *
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
T0*/
_output_shapes
:?????????@@ 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_25_layer_call_fn_5325338

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_53213992
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
5__inference_conv2d_transpose_15_layer_call_fn_5325755

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_53229052
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
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5321317

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
?
?
F__inference_conv2d_13_layer_call_and_return_conditional_losses_5325325

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@0*
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
T0*/
_output_shapes
:?????????@@02	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@02
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_29_layer_call_fn_5326065

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
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_53233872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5322293

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5322784

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
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5326013

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
T0*/
_output_shapes
:?????????@@ *
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
T0*/
_output_shapes
:?????????@@ 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_18_layer_call_fn_5326355

inputs!
unknown:@0
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_53230732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_26_layer_call_fn_5325482

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_53215252
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
?
?
5__inference_conv2d_transpose_17_layer_call_fn_5326155

inputs!
unknown:0 
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_53230172
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
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5323440

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
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5323096

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
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5322079

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
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326283

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
5__inference_conv2d_transpose_17_layer_call_fn_5326146

inputs!
unknown:0 
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_53222212
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
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5323655

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
?
?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325539

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
?
?
5__inference_conv2d_transpose_19_layer_call_fn_5326546

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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_53226492
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
?
?
+__inference_conv2d_14_layer_call_fn_5325458

inputs!
unknown:0@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_53228052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325557

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
?
?
8__inference_batch_normalization_27_layer_call_fn_5325665

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_53234932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5321569

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
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326501

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
?
?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5323601

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
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
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
:?????????@@02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?w
?
D__inference_model_3_layer_call_and_return_conditional_losses_5324233
input_4+
conv2d_12_5324115: 
conv2d_12_5324117: ,
batch_normalization_24_5324120: ,
batch_normalization_24_5324122: ,
batch_normalization_24_5324124: ,
batch_normalization_24_5324126: +
conv2d_13_5324129: 0
conv2d_13_5324131:0,
batch_normalization_25_5324134:0,
batch_normalization_25_5324136:0,
batch_normalization_25_5324138:0,
batch_normalization_25_5324140:0+
conv2d_14_5324143:0@
conv2d_14_5324145:@,
batch_normalization_26_5324148:@,
batch_normalization_26_5324150:@,
batch_normalization_26_5324152:@,
batch_normalization_26_5324154:@+
conv2d_15_5324157:@
conv2d_15_5324159:,
batch_normalization_27_5324162:,
batch_normalization_27_5324164:,
batch_normalization_27_5324166:,
batch_normalization_27_5324168:5
conv2d_transpose_15_5324171:)
conv2d_transpose_15_5324173:,
batch_normalization_28_5324176:,
batch_normalization_28_5324178:,
batch_normalization_28_5324180:,
batch_normalization_28_5324182:5
conv2d_transpose_16_5324185: )
conv2d_transpose_16_5324187: ,
batch_normalization_29_5324190: ,
batch_normalization_29_5324192: ,
batch_normalization_29_5324194: ,
batch_normalization_29_5324196: 5
conv2d_transpose_17_5324199:0 )
conv2d_transpose_17_5324201:0,
batch_normalization_30_5324204:0,
batch_normalization_30_5324206:0,
batch_normalization_30_5324208:0,
batch_normalization_30_5324210:05
conv2d_transpose_18_5324213:@0)
conv2d_transpose_18_5324215:@,
batch_normalization_31_5324218:@,
batch_normalization_31_5324220:@,
batch_normalization_31_5324222:@,
batch_normalization_31_5324224:@5
conv2d_transpose_19_5324227:@)
conv2d_transpose_19_5324229:
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?+conv2d_transpose_16/StatefulPartitionedCall?+conv2d_transpose_17/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_12_5324115conv2d_12_5324117*
Tin
2*
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
GPU2*0J 8? *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_53227172#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_5324120batch_normalization_24_5324122batch_normalization_24_5324124batch_normalization_24_5324126*
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_532274020
.batch_normalization_24/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_13_5324129conv2d_13_5324131*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_53227612#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_5324134batch_normalization_25_5324136batch_normalization_25_5324138batch_normalization_25_5324140*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_532278420
.batch_normalization_25/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_14_5324143conv2d_14_5324145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_53228052#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_26_5324148batch_normalization_26_5324150batch_normalization_26_5324152batch_normalization_26_5324154*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_532282820
.batch_normalization_26/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_15_5324157conv2d_15_5324159*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_53228492#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_27_5324162batch_normalization_27_5324164batch_normalization_27_5324166batch_normalization_27_5324168*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_532287220
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_15_5324171conv2d_transpose_15_5324173*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_53229052-
+conv2d_transpose_15/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0batch_normalization_28_5324176batch_normalization_28_5324178batch_normalization_28_5324180batch_normalization_28_5324182*
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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_532292820
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_16_5324185conv2d_transpose_16_5324187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_53229612-
+conv2d_transpose_16/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0batch_normalization_29_5324190batch_normalization_29_5324192batch_normalization_29_5324194batch_normalization_29_5324196*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_532298420
.batch_normalization_29/StatefulPartitionedCall?
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv2d_transpose_17_5324199conv2d_transpose_17_5324201*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_53230172-
+conv2d_transpose_17/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0batch_normalization_30_5324204batch_normalization_30_5324206batch_normalization_30_5324208batch_normalization_30_5324210*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_532304020
.batch_normalization_30/StatefulPartitionedCall?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv2d_transpose_18_5324213conv2d_transpose_18_5324215*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_53230732-
+conv2d_transpose_18/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0batch_normalization_31_5324218batch_normalization_31_5324220batch_normalization_31_5324222batch_normalization_31_5324224*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_532309620
.batch_normalization_31/StatefulPartitionedCall?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_transpose_19_5324227conv2d_transpose_19_5324229*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_53231292-
+conv2d_transpose_19/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_19/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325449

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
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
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
:?????????@@02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_26_layer_call_fn_5325508

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
:?????????  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_53228282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_24_layer_call_fn_5325220

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_53227402
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
?	
?
8__inference_batch_normalization_31_layer_call_fn_5326439

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
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_53225512
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
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326483

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
?
?
5__inference_conv2d_transpose_18_layer_call_fn_5326346

inputs!
unknown:@0
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_53224352
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
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_27_layer_call_fn_5325639

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_53216952
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
?w
?
D__inference_model_3_layer_call_and_return_conditional_losses_5324354
input_4+
conv2d_12_5324236: 
conv2d_12_5324238: ,
batch_normalization_24_5324241: ,
batch_normalization_24_5324243: ,
batch_normalization_24_5324245: ,
batch_normalization_24_5324247: +
conv2d_13_5324250: 0
conv2d_13_5324252:0,
batch_normalization_25_5324255:0,
batch_normalization_25_5324257:0,
batch_normalization_25_5324259:0,
batch_normalization_25_5324261:0+
conv2d_14_5324264:0@
conv2d_14_5324266:@,
batch_normalization_26_5324269:@,
batch_normalization_26_5324271:@,
batch_normalization_26_5324273:@,
batch_normalization_26_5324275:@+
conv2d_15_5324278:@
conv2d_15_5324280:,
batch_normalization_27_5324283:,
batch_normalization_27_5324285:,
batch_normalization_27_5324287:,
batch_normalization_27_5324289:5
conv2d_transpose_15_5324292:)
conv2d_transpose_15_5324294:,
batch_normalization_28_5324297:,
batch_normalization_28_5324299:,
batch_normalization_28_5324301:,
batch_normalization_28_5324303:5
conv2d_transpose_16_5324306: )
conv2d_transpose_16_5324308: ,
batch_normalization_29_5324311: ,
batch_normalization_29_5324313: ,
batch_normalization_29_5324315: ,
batch_normalization_29_5324317: 5
conv2d_transpose_17_5324320:0 )
conv2d_transpose_17_5324322:0,
batch_normalization_30_5324325:0,
batch_normalization_30_5324327:0,
batch_normalization_30_5324329:0,
batch_normalization_30_5324331:05
conv2d_transpose_18_5324334:@0)
conv2d_transpose_18_5324336:@,
batch_normalization_31_5324339:@,
batch_normalization_31_5324341:@,
batch_normalization_31_5324343:@,
batch_normalization_31_5324345:@5
conv2d_transpose_19_5324348:@)
conv2d_transpose_19_5324350:
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?+conv2d_transpose_16/StatefulPartitionedCall?+conv2d_transpose_17/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_12_5324236conv2d_12_5324238*
Tin
2*
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
GPU2*0J 8? *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_53227172#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_5324241batch_normalization_24_5324243batch_normalization_24_5324245batch_normalization_24_5324247*
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_532365520
.batch_normalization_24/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_13_5324250conv2d_13_5324252*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_53227612#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_5324255batch_normalization_25_5324257batch_normalization_25_5324259batch_normalization_25_5324261*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_532360120
.batch_normalization_25/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_14_5324264conv2d_14_5324266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_53228052#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_26_5324269batch_normalization_26_5324271batch_normalization_26_5324273batch_normalization_26_5324275*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_532354720
.batch_normalization_26/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_15_5324278conv2d_15_5324280*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_53228492#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_27_5324283batch_normalization_27_5324285batch_normalization_27_5324287batch_normalization_27_5324289*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_532349320
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_15_5324292conv2d_transpose_15_5324294*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_53229052-
+conv2d_transpose_15/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0batch_normalization_28_5324297batch_normalization_28_5324299batch_normalization_28_5324301batch_normalization_28_5324303*
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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_532344020
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_16_5324306conv2d_transpose_16_5324308*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_53229612-
+conv2d_transpose_16/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0batch_normalization_29_5324311batch_normalization_29_5324313batch_normalization_29_5324315batch_normalization_29_5324317*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_532338720
.batch_normalization_29/StatefulPartitionedCall?
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv2d_transpose_17_5324320conv2d_transpose_17_5324322*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_53230172-
+conv2d_transpose_17/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0batch_normalization_30_5324325batch_normalization_30_5324327batch_normalization_30_5324329batch_normalization_30_5324331*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_532333420
.batch_normalization_30/StatefulPartitionedCall?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv2d_transpose_18_5324334conv2d_transpose_18_5324336*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_53230732-
+conv2d_transpose_18/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0batch_normalization_31_5324339batch_normalization_31_5324341batch_normalization_31_5324343batch_normalization_31_5324345*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_532328120
.batch_normalization_31/StatefulPartitionedCall?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_transpose_19_5324348conv2d_transpose_19_5324350*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_53231292-
+conv2d_transpose_19/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_19/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5326613

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
:???????????*
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
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoidp
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5323073

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@
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
:???????????@*
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
:???????????@2	
BiasAddb
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
F__inference_conv2d_12_layer_call_and_return_conditional_losses_5325181

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
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
:??????????? 2	
BiasAddb
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_model_3_layer_call_fn_5324568

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35:0 

unknown_36:0

unknown_37:0

unknown_38:0

unknown_39:0

unknown_40:0$

unknown_41:@0

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@

unknown_48:
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_53231362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?%
?
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5326189

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_30_layer_call_fn_5326252

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_53230402
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
?
?
5__inference_conv2d_transpose_19_layer_call_fn_5326555

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
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_53231292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?%
?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5322435

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_25_layer_call_fn_5325364

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
 */
_output_shapes
:?????????@@0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_53227842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
??
?/
D__inference_model_3_layer_call_and_return_conditional_losses_5324917

inputsB
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: <
.batch_normalization_24_readvariableop_resource: >
0batch_normalization_24_readvariableop_1_resource: M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_13_conv2d_readvariableop_resource: 07
)conv2d_13_biasadd_readvariableop_resource:0<
.batch_normalization_25_readvariableop_resource:0>
0batch_normalization_25_readvariableop_1_resource:0M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource:0O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:0B
(conv2d_14_conv2d_readvariableop_resource:0@7
)conv2d_14_biasadd_readvariableop_resource:@<
.batch_normalization_26_readvariableop_resource:@>
0batch_normalization_26_readvariableop_1_resource:@M
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_15_conv2d_readvariableop_resource:@7
)conv2d_15_biasadd_readvariableop_resource:<
.batch_normalization_27_readvariableop_resource:>
0batch_normalization_27_readvariableop_1_resource:M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_15_biasadd_readvariableop_resource:<
.batch_normalization_28_readvariableop_resource:>
0batch_normalization_28_readvariableop_1_resource:M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:V
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_16_biasadd_readvariableop_resource: <
.batch_normalization_29_readvariableop_resource: >
0batch_normalization_29_readvariableop_1_resource: M
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource:0 A
3conv2d_transpose_17_biasadd_readvariableop_resource:0<
.batch_normalization_30_readvariableop_resource:0>
0batch_normalization_30_readvariableop_1_resource:0M
?batch_normalization_30_fusedbatchnormv3_readvariableop_resource:0O
Abatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:0V
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource:@0A
3conv2d_transpose_18_biasadd_readvariableop_resource:@<
.batch_normalization_31_readvariableop_resource:@>
0batch_normalization_31_readvariableop_1_resource:@M
?batch_normalization_31_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_19_biasadd_readvariableop_resource:
identity??6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_24/ReadVariableOp?'batch_normalization_24/ReadVariableOp_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1?6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1?6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_30/ReadVariableOp?'batch_normalization_30/ReadVariableOp_1?6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_31/ReadVariableOp?'batch_normalization_31/ReadVariableOp_1? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?*conv2d_transpose_16/BiasAdd/ReadVariableOp?3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?*conv2d_transpose_17/BiasAdd/ReadVariableOp?3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_12/BiasAdd?
conv2d_12/TanhTanhconv2d_12/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_12/Tanh?
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_24/ReadVariableOp?
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_24/ReadVariableOp_1?
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_12/Tanh:y:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_24/FusedBatchNormV3?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@0*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@02
conv2d_13/BiasAdd~
conv2d_13/TanhTanhconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@02
conv2d_13/Tanh?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:0*
dtype02'
%batch_normalization_25/ReadVariableOp?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:0*
dtype02)
'batch_normalization_25/ReadVariableOp_1?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_13/Tanh:y:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
epsilon%o?:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_14/BiasAdd~
conv2d_14/TanhTanhconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_14/Tanh?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3conv2d_14/Tanh:y:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_15/BiasAdd~
conv2d_15/TanhTanhconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_15/Tanh?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_15/Tanh:y:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3?
conv2d_transpose_15/ShapeShape+batch_normalization_27/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape?
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack?
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1?
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_15/stack/2|
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/3?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack?
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stack?
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1?
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transpose?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_15/BiasAdd/ReadVariableOp?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_15/BiasAdd?
conv2d_transpose_15/TanhTanh$conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_15/Tanh?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_15/Tanh:y:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_28/FusedBatchNormV3?
conv2d_transpose_16/ShapeShape+batch_normalization_28/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape?
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack?
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1?
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2?
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_16/stack/2|
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_16/stack/3?
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack?
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stack?
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1?
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2?
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1?
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transpose?
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOp?
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_16/BiasAdd?
conv2d_transpose_16/TanhTanh$conv2d_transpose_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_16/Tanh?
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_29/ReadVariableOp?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_29/ReadVariableOp_1?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_16/Tanh:y:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_29/FusedBatchNormV3?
conv2d_transpose_17/ShapeShape+batch_normalization_29/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape?
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack?
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1?
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2?
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice}
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_17/stack/1}
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_17/stack/2|
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_17/stack/3?
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack?
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stack?
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1?
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2?
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1?
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_29/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transpose?
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOp?
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_transpose_17/BiasAdd?
conv2d_transpose_17/TanhTanh$conv2d_transpose_17/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_transpose_17/Tanh?
%batch_normalization_30/ReadVariableOpReadVariableOp.batch_normalization_30_readvariableop_resource*
_output_shapes
:0*
dtype02'
%batch_normalization_30/ReadVariableOp?
'batch_normalization_30/ReadVariableOp_1ReadVariableOp0batch_normalization_30_readvariableop_1_resource*
_output_shapes
:0*
dtype02)
'batch_normalization_30/ReadVariableOp_1?
6batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype028
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02:
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_30/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_17/Tanh:y:0-batch_normalization_30/ReadVariableOp:value:0/batch_normalization_30/ReadVariableOp_1:value:0>batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2)
'batch_normalization_30/FusedBatchNormV3?
conv2d_transpose_18/ShapeShape+batch_normalization_30/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_18/Shape?
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_18/strided_slice/stack?
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_1?
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_2?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_18/strided_slice}
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_18/stack/1}
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_18/stack/2|
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_18/stack/3?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_18/stack?
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_18/strided_slice_1/stack?
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_1?
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_2?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_18/strided_slice_1?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype025
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_30/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2&
$conv2d_transpose_18/conv2d_transpose?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_18/BiasAdd/ReadVariableOp?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_18/BiasAdd?
conv2d_transpose_18/TanhTanh$conv2d_transpose_18/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_18/Tanh?
%batch_normalization_31/ReadVariableOpReadVariableOp.batch_normalization_31_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_31/ReadVariableOp?
'batch_normalization_31/ReadVariableOp_1ReadVariableOp0batch_normalization_31_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_31/ReadVariableOp_1?
6batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_31/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_18/Tanh:y:0-batch_normalization_31/ReadVariableOp:value:0/batch_normalization_31/ReadVariableOp_1:value:0>batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_31/FusedBatchNormV3?
conv2d_transpose_19/ShapeShape+batch_normalization_31/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_19/Shape?
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_19/strided_slice/stack?
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_1?
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_2?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_19/strided_slice}
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_19/stack/1}
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_19/stack/2|
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_19/stack/3?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_19/stack?
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_19/strided_slice_1/stack?
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_1?
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_2?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_19/strided_slice_1?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_31/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$conv2d_transpose_19/conv2d_transpose?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_19/BiasAdd/ReadVariableOp?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_19/BiasAdd?
conv2d_transpose_19/SigmoidSigmoid$conv2d_transpose_19/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_19/Sigmoid?
IdentityIdentityconv2d_transpose_19/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp7^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_17^batch_normalization_30/FusedBatchNormV3/ReadVariableOp9^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_30/ReadVariableOp(^batch_normalization_30/ReadVariableOp_17^batch_normalization_31/FusedBatchNormV3/ReadVariableOp9^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_31/ReadVariableOp(^batch_normalization_31/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12p
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp6batch_normalization_30/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_18batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_30/ReadVariableOp%batch_normalization_30/ReadVariableOp2R
'batch_normalization_30/ReadVariableOp_1'batch_normalization_30/ReadVariableOp_12p
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp6batch_normalization_31/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_18batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_31/ReadVariableOp%batch_normalization_31/ReadVariableOp2R
'batch_normalization_31/ReadVariableOp_1'batch_normalization_31/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325683

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5322828

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
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_27_layer_call_fn_5325626

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_53216512
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
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5323387

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
U0*K
_output_shapes9
7:?????????@@ : : : : :*
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
:?????????@@ 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326319

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
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5322337

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
?	
?
8__inference_batch_normalization_29_layer_call_fn_5326039

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
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_53221232
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
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325737

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
7:?????????:::::*
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
:?????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325901

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
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5322123

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
??
?5
"__inference__wrapped_model_5321251
input_4J
0model_3_conv2d_12_conv2d_readvariableop_resource: ?
1model_3_conv2d_12_biasadd_readvariableop_resource: D
6model_3_batch_normalization_24_readvariableop_resource: F
8model_3_batch_normalization_24_readvariableop_1_resource: U
Gmodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_resource: W
Imodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource: J
0model_3_conv2d_13_conv2d_readvariableop_resource: 0?
1model_3_conv2d_13_biasadd_readvariableop_resource:0D
6model_3_batch_normalization_25_readvariableop_resource:0F
8model_3_batch_normalization_25_readvariableop_1_resource:0U
Gmodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_resource:0W
Imodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:0J
0model_3_conv2d_14_conv2d_readvariableop_resource:0@?
1model_3_conv2d_14_biasadd_readvariableop_resource:@D
6model_3_batch_normalization_26_readvariableop_resource:@F
8model_3_batch_normalization_26_readvariableop_1_resource:@U
Gmodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:@W
Imodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:@J
0model_3_conv2d_15_conv2d_readvariableop_resource:@?
1model_3_conv2d_15_biasadd_readvariableop_resource:D
6model_3_batch_normalization_27_readvariableop_resource:F
8model_3_batch_normalization_27_readvariableop_1_resource:U
Gmodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:W
Imodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:^
Dmodel_3_conv2d_transpose_15_conv2d_transpose_readvariableop_resource:I
;model_3_conv2d_transpose_15_biasadd_readvariableop_resource:D
6model_3_batch_normalization_28_readvariableop_resource:F
8model_3_batch_normalization_28_readvariableop_1_resource:U
Gmodel_3_batch_normalization_28_fusedbatchnormv3_readvariableop_resource:W
Imodel_3_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:^
Dmodel_3_conv2d_transpose_16_conv2d_transpose_readvariableop_resource: I
;model_3_conv2d_transpose_16_biasadd_readvariableop_resource: D
6model_3_batch_normalization_29_readvariableop_resource: F
8model_3_batch_normalization_29_readvariableop_1_resource: U
Gmodel_3_batch_normalization_29_fusedbatchnormv3_readvariableop_resource: W
Imodel_3_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource: ^
Dmodel_3_conv2d_transpose_17_conv2d_transpose_readvariableop_resource:0 I
;model_3_conv2d_transpose_17_biasadd_readvariableop_resource:0D
6model_3_batch_normalization_30_readvariableop_resource:0F
8model_3_batch_normalization_30_readvariableop_1_resource:0U
Gmodel_3_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:0W
Imodel_3_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:0^
Dmodel_3_conv2d_transpose_18_conv2d_transpose_readvariableop_resource:@0I
;model_3_conv2d_transpose_18_biasadd_readvariableop_resource:@D
6model_3_batch_normalization_31_readvariableop_resource:@F
8model_3_batch_normalization_31_readvariableop_1_resource:@U
Gmodel_3_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:@W
Imodel_3_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:@^
Dmodel_3_conv2d_transpose_19_conv2d_transpose_readvariableop_resource:@I
;model_3_conv2d_transpose_19_biasadd_readvariableop_resource:
identity??>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_24/ReadVariableOp?/model_3/batch_normalization_24/ReadVariableOp_1?>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_25/ReadVariableOp?/model_3/batch_normalization_25/ReadVariableOp_1?>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_26/ReadVariableOp?/model_3/batch_normalization_26/ReadVariableOp_1?>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_27/ReadVariableOp?/model_3/batch_normalization_27/ReadVariableOp_1?>model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_28/ReadVariableOp?/model_3/batch_normalization_28/ReadVariableOp_1?>model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_29/ReadVariableOp?/model_3/batch_normalization_29/ReadVariableOp_1?>model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_30/ReadVariableOp?/model_3/batch_normalization_30/ReadVariableOp_1?>model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_31/ReadVariableOp?/model_3/batch_normalization_31/ReadVariableOp_1?(model_3/conv2d_12/BiasAdd/ReadVariableOp?'model_3/conv2d_12/Conv2D/ReadVariableOp?(model_3/conv2d_13/BiasAdd/ReadVariableOp?'model_3/conv2d_13/Conv2D/ReadVariableOp?(model_3/conv2d_14/BiasAdd/ReadVariableOp?'model_3/conv2d_14/Conv2D/ReadVariableOp?(model_3/conv2d_15/BiasAdd/ReadVariableOp?'model_3/conv2d_15/Conv2D/ReadVariableOp?2model_3/conv2d_transpose_15/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
'model_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_12/Conv2D/ReadVariableOp?
model_3/conv2d_12/Conv2DConv2Dinput_4/model_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_3/conv2d_12/Conv2D?
(model_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_12/BiasAdd/ReadVariableOp?
model_3/conv2d_12/BiasAddBiasAdd!model_3/conv2d_12/Conv2D:output:00model_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_12/BiasAdd?
model_3/conv2d_12/TanhTanh"model_3/conv2d_12/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_12/Tanh?
-model_3/batch_normalization_24/ReadVariableOpReadVariableOp6model_3_batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_3/batch_normalization_24/ReadVariableOp?
/model_3/batch_normalization_24/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_3/batch_normalization_24/ReadVariableOp_1?
>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3model_3/conv2d_12/Tanh:y:05model_3/batch_normalization_24/ReadVariableOp:value:07model_3/batch_normalization_24/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_24/FusedBatchNormV3?
'model_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02)
'model_3/conv2d_13/Conv2D/ReadVariableOp?
model_3/conv2d_13/Conv2DConv2D3model_3/batch_normalization_24/FusedBatchNormV3:y:0/model_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@0*
paddingSAME*
strides
2
model_3/conv2d_13/Conv2D?
(model_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(model_3/conv2d_13/BiasAdd/ReadVariableOp?
model_3/conv2d_13/BiasAddBiasAdd!model_3/conv2d_13/Conv2D:output:00model_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@02
model_3/conv2d_13/BiasAdd?
model_3/conv2d_13/TanhTanh"model_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@02
model_3/conv2d_13/Tanh?
-model_3/batch_normalization_25/ReadVariableOpReadVariableOp6model_3_batch_normalization_25_readvariableop_resource*
_output_shapes
:0*
dtype02/
-model_3/batch_normalization_25/ReadVariableOp?
/model_3/batch_normalization_25/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_25_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/model_3/batch_normalization_25/ReadVariableOp_1?
>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3model_3/conv2d_13/Tanh:y:05model_3/batch_normalization_25/ReadVariableOp:value:07model_3/batch_normalization_25/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_25/FusedBatchNormV3?
'model_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02)
'model_3/conv2d_14/Conv2D/ReadVariableOp?
model_3/conv2d_14/Conv2DConv2D3model_3/batch_normalization_25/FusedBatchNormV3:y:0/model_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
model_3/conv2d_14/Conv2D?
(model_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_14/BiasAdd/ReadVariableOp?
model_3/conv2d_14/BiasAddBiasAdd!model_3/conv2d_14/Conv2D:output:00model_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
model_3/conv2d_14/BiasAdd?
model_3/conv2d_14/TanhTanh"model_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
model_3/conv2d_14/Tanh?
-model_3/batch_normalization_26/ReadVariableOpReadVariableOp6model_3_batch_normalization_26_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_3/batch_normalization_26/ReadVariableOp?
/model_3/batch_normalization_26/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_26_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_3/batch_normalization_26/ReadVariableOp_1?
>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3model_3/conv2d_14/Tanh:y:05model_3/batch_normalization_26/ReadVariableOp:value:07model_3/batch_normalization_26/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_26/FusedBatchNormV3?
'model_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'model_3/conv2d_15/Conv2D/ReadVariableOp?
model_3/conv2d_15/Conv2DConv2D3model_3/batch_normalization_26/FusedBatchNormV3:y:0/model_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_3/conv2d_15/Conv2D?
(model_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_15/BiasAdd/ReadVariableOp?
model_3/conv2d_15/BiasAddBiasAdd!model_3/conv2d_15/Conv2D:output:00model_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_3/conv2d_15/BiasAdd?
model_3/conv2d_15/TanhTanh"model_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_3/conv2d_15/Tanh?
-model_3/batch_normalization_27/ReadVariableOpReadVariableOp6model_3_batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_3/batch_normalization_27/ReadVariableOp?
/model_3/batch_normalization_27/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_3/batch_normalization_27/ReadVariableOp_1?
>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3model_3/conv2d_15/Tanh:y:05model_3/batch_normalization_27/ReadVariableOp:value:07model_3/batch_normalization_27/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_27/FusedBatchNormV3?
!model_3/conv2d_transpose_15/ShapeShape3model_3/batch_normalization_27/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_15/Shape?
/model_3/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_15/strided_slice/stack?
1model_3/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_15/strided_slice/stack_1?
1model_3/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_15/strided_slice/stack_2?
)model_3/conv2d_transpose_15/strided_sliceStridedSlice*model_3/conv2d_transpose_15/Shape:output:08model_3/conv2d_transpose_15/strided_slice/stack:output:0:model_3/conv2d_transpose_15/strided_slice/stack_1:output:0:model_3/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_15/strided_slice?
#model_3/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_3/conv2d_transpose_15/stack/1?
#model_3/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_3/conv2d_transpose_15/stack/2?
#model_3/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_3/conv2d_transpose_15/stack/3?
!model_3/conv2d_transpose_15/stackPack2model_3/conv2d_transpose_15/strided_slice:output:0,model_3/conv2d_transpose_15/stack/1:output:0,model_3/conv2d_transpose_15/stack/2:output:0,model_3/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_15/stack?
1model_3/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_15/strided_slice_1/stack?
3model_3/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_15/strided_slice_1/stack_1?
3model_3/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_15/strided_slice_1/stack_2?
+model_3/conv2d_transpose_15/strided_slice_1StridedSlice*model_3/conv2d_transpose_15/stack:output:0:model_3/conv2d_transpose_15/strided_slice_1/stack:output:0<model_3/conv2d_transpose_15/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_15/strided_slice_1?
;model_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;model_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
,model_3/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_15/stack:output:0Cmodel_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:03model_3/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2.
,model_3/conv2d_transpose_15/conv2d_transpose?
2model_3/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2model_3/conv2d_transpose_15/BiasAdd/ReadVariableOp?
#model_3/conv2d_transpose_15/BiasAddBiasAdd5model_3/conv2d_transpose_15/conv2d_transpose:output:0:model_3/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2%
#model_3/conv2d_transpose_15/BiasAdd?
 model_3/conv2d_transpose_15/TanhTanh,model_3/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2"
 model_3/conv2d_transpose_15/Tanh?
-model_3/batch_normalization_28/ReadVariableOpReadVariableOp6model_3_batch_normalization_28_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_3/batch_normalization_28/ReadVariableOp?
/model_3/batch_normalization_28/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_28_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_3/batch_normalization_28/ReadVariableOp_1?
>model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_transpose_15/Tanh:y:05model_3/batch_normalization_28/ReadVariableOp:value:07model_3/batch_normalization_28/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_28/FusedBatchNormV3?
!model_3/conv2d_transpose_16/ShapeShape3model_3/batch_normalization_28/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_16/Shape?
/model_3/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_16/strided_slice/stack?
1model_3/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_16/strided_slice/stack_1?
1model_3/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_16/strided_slice/stack_2?
)model_3/conv2d_transpose_16/strided_sliceStridedSlice*model_3/conv2d_transpose_16/Shape:output:08model_3/conv2d_transpose_16/strided_slice/stack:output:0:model_3/conv2d_transpose_16/strided_slice/stack_1:output:0:model_3/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_16/strided_slice?
#model_3/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#model_3/conv2d_transpose_16/stack/1?
#model_3/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2%
#model_3/conv2d_transpose_16/stack/2?
#model_3/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_3/conv2d_transpose_16/stack/3?
!model_3/conv2d_transpose_16/stackPack2model_3/conv2d_transpose_16/strided_slice:output:0,model_3/conv2d_transpose_16/stack/1:output:0,model_3/conv2d_transpose_16/stack/2:output:0,model_3/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_16/stack?
1model_3/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_16/strided_slice_1/stack?
3model_3/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_16/strided_slice_1/stack_1?
3model_3/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_16/strided_slice_1/stack_2?
+model_3/conv2d_transpose_16/strided_slice_1StridedSlice*model_3/conv2d_transpose_16/stack:output:0:model_3/conv2d_transpose_16/strided_slice_1/stack:output:0<model_3/conv2d_transpose_16/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_16/strided_slice_1?
;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
,model_3/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_16/stack:output:0Cmodel_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:03model_3/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2.
,model_3/conv2d_transpose_16/conv2d_transpose?
2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp?
#model_3/conv2d_transpose_16/BiasAddBiasAdd5model_3/conv2d_transpose_16/conv2d_transpose:output:0:model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2%
#model_3/conv2d_transpose_16/BiasAdd?
 model_3/conv2d_transpose_16/TanhTanh,model_3/conv2d_transpose_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2"
 model_3/conv2d_transpose_16/Tanh?
-model_3/batch_normalization_29/ReadVariableOpReadVariableOp6model_3_batch_normalization_29_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_3/batch_normalization_29/ReadVariableOp?
/model_3/batch_normalization_29/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_29_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_3/batch_normalization_29/ReadVariableOp_1?
>model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_29/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_transpose_16/Tanh:y:05model_3/batch_normalization_29/ReadVariableOp:value:07model_3/batch_normalization_29/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_29/FusedBatchNormV3?
!model_3/conv2d_transpose_17/ShapeShape3model_3/batch_normalization_29/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_17/Shape?
/model_3/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_17/strided_slice/stack?
1model_3/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_17/strided_slice/stack_1?
1model_3/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_17/strided_slice/stack_2?
)model_3/conv2d_transpose_17/strided_sliceStridedSlice*model_3/conv2d_transpose_17/Shape:output:08model_3/conv2d_transpose_17/strided_slice/stack:output:0:model_3/conv2d_transpose_17/strided_slice/stack_1:output:0:model_3/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_17/strided_slice?
#model_3/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_17/stack/1?
#model_3/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_17/stack/2?
#model_3/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02%
#model_3/conv2d_transpose_17/stack/3?
!model_3/conv2d_transpose_17/stackPack2model_3/conv2d_transpose_17/strided_slice:output:0,model_3/conv2d_transpose_17/stack/1:output:0,model_3/conv2d_transpose_17/stack/2:output:0,model_3/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_17/stack?
1model_3/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_17/strided_slice_1/stack?
3model_3/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_17/strided_slice_1/stack_1?
3model_3/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_17/strided_slice_1/stack_2?
+model_3/conv2d_transpose_17/strided_slice_1StridedSlice*model_3/conv2d_transpose_17/stack:output:0:model_3/conv2d_transpose_17/strided_slice_1/stack:output:0<model_3/conv2d_transpose_17/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_17/strided_slice_1?
;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype02=
;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
,model_3/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_17/stack:output:0Cmodel_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:03model_3/batch_normalization_29/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2.
,model_3/conv2d_transpose_17/conv2d_transpose?
2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype024
2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp?
#model_3/conv2d_transpose_17/BiasAddBiasAdd5model_3/conv2d_transpose_17/conv2d_transpose:output:0:model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02%
#model_3/conv2d_transpose_17/BiasAdd?
 model_3/conv2d_transpose_17/TanhTanh,model_3/conv2d_transpose_17/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02"
 model_3/conv2d_transpose_17/Tanh?
-model_3/batch_normalization_30/ReadVariableOpReadVariableOp6model_3_batch_normalization_30_readvariableop_resource*
_output_shapes
:0*
dtype02/
-model_3/batch_normalization_30/ReadVariableOp?
/model_3/batch_normalization_30/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_30_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/model_3/batch_normalization_30/ReadVariableOp_1?
>model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_30/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_transpose_17/Tanh:y:05model_3/batch_normalization_30/ReadVariableOp:value:07model_3/batch_normalization_30/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_30/FusedBatchNormV3?
!model_3/conv2d_transpose_18/ShapeShape3model_3/batch_normalization_30/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_18/Shape?
/model_3/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_18/strided_slice/stack?
1model_3/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_18/strided_slice/stack_1?
1model_3/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_18/strided_slice/stack_2?
)model_3/conv2d_transpose_18/strided_sliceStridedSlice*model_3/conv2d_transpose_18/Shape:output:08model_3/conv2d_transpose_18/strided_slice/stack:output:0:model_3/conv2d_transpose_18/strided_slice/stack_1:output:0:model_3/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_18/strided_slice?
#model_3/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_18/stack/1?
#model_3/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_18/stack/2?
#model_3/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2%
#model_3/conv2d_transpose_18/stack/3?
!model_3/conv2d_transpose_18/stackPack2model_3/conv2d_transpose_18/strided_slice:output:0,model_3/conv2d_transpose_18/stack/1:output:0,model_3/conv2d_transpose_18/stack/2:output:0,model_3/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_18/stack?
1model_3/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_18/strided_slice_1/stack?
3model_3/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_18/strided_slice_1/stack_1?
3model_3/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_18/strided_slice_1/stack_2?
+model_3/conv2d_transpose_18/strided_slice_1StridedSlice*model_3/conv2d_transpose_18/stack:output:0:model_3/conv2d_transpose_18/strided_slice_1/stack:output:0<model_3/conv2d_transpose_18/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_18/strided_slice_1?
;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype02=
;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
,model_3/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_18/stack:output:0Cmodel_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:03model_3/batch_normalization_30/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2.
,model_3/conv2d_transpose_18/conv2d_transpose?
2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp?
#model_3/conv2d_transpose_18/BiasAddBiasAdd5model_3/conv2d_transpose_18/conv2d_transpose:output:0:model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2%
#model_3/conv2d_transpose_18/BiasAdd?
 model_3/conv2d_transpose_18/TanhTanh,model_3/conv2d_transpose_18/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2"
 model_3/conv2d_transpose_18/Tanh?
-model_3/batch_normalization_31/ReadVariableOpReadVariableOp6model_3_batch_normalization_31_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_3/batch_normalization_31/ReadVariableOp?
/model_3/batch_normalization_31/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_3/batch_normalization_31/ReadVariableOp_1?
>model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_31/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_transpose_18/Tanh:y:05model_3/batch_normalization_31/ReadVariableOp:value:07model_3/batch_normalization_31/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_31/FusedBatchNormV3?
!model_3/conv2d_transpose_19/ShapeShape3model_3/batch_normalization_31/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_19/Shape?
/model_3/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_19/strided_slice/stack?
1model_3/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_19/strided_slice/stack_1?
1model_3/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_19/strided_slice/stack_2?
)model_3/conv2d_transpose_19/strided_sliceStridedSlice*model_3/conv2d_transpose_19/Shape:output:08model_3/conv2d_transpose_19/strided_slice/stack:output:0:model_3/conv2d_transpose_19/strided_slice/stack_1:output:0:model_3/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_19/strided_slice?
#model_3/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_19/stack/1?
#model_3/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_19/stack/2?
#model_3/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_3/conv2d_transpose_19/stack/3?
!model_3/conv2d_transpose_19/stackPack2model_3/conv2d_transpose_19/strided_slice:output:0,model_3/conv2d_transpose_19/stack/1:output:0,model_3/conv2d_transpose_19/stack/2:output:0,model_3/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_19/stack?
1model_3/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_19/strided_slice_1/stack?
3model_3/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_19/strided_slice_1/stack_1?
3model_3/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_19/strided_slice_1/stack_2?
+model_3/conv2d_transpose_19/strided_slice_1StridedSlice*model_3/conv2d_transpose_19/stack:output:0:model_3/conv2d_transpose_19/strided_slice_1/stack:output:0<model_3/conv2d_transpose_19/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_19/strided_slice_1?
;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02=
;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
,model_3/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_19/stack:output:0Cmodel_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:03model_3/batch_normalization_31/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2.
,model_3/conv2d_transpose_19/conv2d_transpose?
2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp?
#model_3/conv2d_transpose_19/BiasAddBiasAdd5model_3/conv2d_transpose_19/conv2d_transpose:output:0:model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2%
#model_3/conv2d_transpose_19/BiasAdd?
#model_3/conv2d_transpose_19/SigmoidSigmoid,model_3/conv2d_transpose_19/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2%
#model_3/conv2d_transpose_19/Sigmoid?
IdentityIdentity'model_3/conv2d_transpose_19/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp?^model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_24/ReadVariableOp0^model_3/batch_normalization_24/ReadVariableOp_1?^model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_25/ReadVariableOp0^model_3/batch_normalization_25/ReadVariableOp_1?^model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_26/ReadVariableOp0^model_3/batch_normalization_26/ReadVariableOp_1?^model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_27/ReadVariableOp0^model_3/batch_normalization_27/ReadVariableOp_1?^model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_28/ReadVariableOp0^model_3/batch_normalization_28/ReadVariableOp_1?^model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_29/ReadVariableOp0^model_3/batch_normalization_29/ReadVariableOp_1?^model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_30/ReadVariableOp0^model_3/batch_normalization_30/ReadVariableOp_1?^model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_31/ReadVariableOp0^model_3/batch_normalization_31/ReadVariableOp_1)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp)^model_3/conv2d_14/BiasAdd/ReadVariableOp(^model_3/conv2d_14/Conv2D/ReadVariableOp)^model_3/conv2d_15/BiasAdd/ReadVariableOp(^model_3/conv2d_15/Conv2D/ReadVariableOp3^model_3/conv2d_transpose_15/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOp3^model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp3^model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp3^model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp3^model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_24/ReadVariableOp-model_3/batch_normalization_24/ReadVariableOp2b
/model_3/batch_normalization_24/ReadVariableOp_1/model_3/batch_normalization_24/ReadVariableOp_12?
>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_25/ReadVariableOp-model_3/batch_normalization_25/ReadVariableOp2b
/model_3/batch_normalization_25/ReadVariableOp_1/model_3/batch_normalization_25/ReadVariableOp_12?
>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_26/ReadVariableOp-model_3/batch_normalization_26/ReadVariableOp2b
/model_3/batch_normalization_26/ReadVariableOp_1/model_3/batch_normalization_26/ReadVariableOp_12?
>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_27/ReadVariableOp-model_3/batch_normalization_27/ReadVariableOp2b
/model_3/batch_normalization_27/ReadVariableOp_1/model_3/batch_normalization_27/ReadVariableOp_12?
>model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_28/ReadVariableOp-model_3/batch_normalization_28/ReadVariableOp2b
/model_3/batch_normalization_28/ReadVariableOp_1/model_3/batch_normalization_28/ReadVariableOp_12?
>model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_29/ReadVariableOp-model_3/batch_normalization_29/ReadVariableOp2b
/model_3/batch_normalization_29/ReadVariableOp_1/model_3/batch_normalization_29/ReadVariableOp_12?
>model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_30/ReadVariableOp-model_3/batch_normalization_30/ReadVariableOp2b
/model_3/batch_normalization_30/ReadVariableOp_1/model_3/batch_normalization_30/ReadVariableOp_12?
>model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_31/ReadVariableOp-model_3/batch_normalization_31/ReadVariableOp2b
/model_3/batch_normalization_31/ReadVariableOp_1/model_3/batch_normalization_31/ReadVariableOp_12T
(model_3/conv2d_12/BiasAdd/ReadVariableOp(model_3/conv2d_12/BiasAdd/ReadVariableOp2R
'model_3/conv2d_12/Conv2D/ReadVariableOp'model_3/conv2d_12/Conv2D/ReadVariableOp2T
(model_3/conv2d_13/BiasAdd/ReadVariableOp(model_3/conv2d_13/BiasAdd/ReadVariableOp2R
'model_3/conv2d_13/Conv2D/ReadVariableOp'model_3/conv2d_13/Conv2D/ReadVariableOp2T
(model_3/conv2d_14/BiasAdd/ReadVariableOp(model_3/conv2d_14/BiasAdd/ReadVariableOp2R
'model_3/conv2d_14/Conv2D/ReadVariableOp'model_3/conv2d_14/Conv2D/ReadVariableOp2T
(model_3/conv2d_15/BiasAdd/ReadVariableOp(model_3/conv2d_15/BiasAdd/ReadVariableOp2R
'model_3/conv2d_15/Conv2D/ReadVariableOp'model_3/conv2d_15/Conv2D/ReadVariableOp2h
2model_3/conv2d_transpose_15/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_15/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2h
2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2h
2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2h
2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2h
2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?z
?
 __inference__traced_save_5326825
file_prefix/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_15_kernel_read_readvariableop7
3savev2_conv2d_transpose_15_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_16_kernel_read_readvariableop7
3savev2_conv2d_transpose_16_bias_read_readvariableop;
7savev2_batch_normalization_29_gamma_read_readvariableop:
6savev2_batch_normalization_29_beta_read_readvariableopA
=savev2_batch_normalization_29_moving_mean_read_readvariableopE
Asavev2_batch_normalization_29_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_17_kernel_read_readvariableop7
3savev2_conv2d_transpose_17_bias_read_readvariableop;
7savev2_batch_normalization_30_gamma_read_readvariableop:
6savev2_batch_normalization_30_beta_read_readvariableopA
=savev2_batch_normalization_30_moving_mean_read_readvariableopE
Asavev2_batch_normalization_30_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_18_kernel_read_readvariableop7
3savev2_conv2d_transpose_18_bias_read_readvariableop;
7savev2_batch_normalization_31_gamma_read_readvariableop:
6savev2_batch_normalization_31_beta_read_readvariableopA
=savev2_batch_normalization_31_moving_mean_read_readvariableopE
Asavev2_batch_normalization_31_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_19_kernel_read_readvariableop7
3savev2_conv2d_transpose_19_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop
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
:@*
dtype0*?
value?B?@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop5savev2_conv2d_transpose_15_kernel_read_readvariableop3savev2_conv2d_transpose_15_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop5savev2_conv2d_transpose_16_kernel_read_readvariableop3savev2_conv2d_transpose_16_bias_read_readvariableop7savev2_batch_normalization_29_gamma_read_readvariableop6savev2_batch_normalization_29_beta_read_readvariableop=savev2_batch_normalization_29_moving_mean_read_readvariableopAsavev2_batch_normalization_29_moving_variance_read_readvariableop5savev2_conv2d_transpose_17_kernel_read_readvariableop3savev2_conv2d_transpose_17_bias_read_readvariableop7savev2_batch_normalization_30_gamma_read_readvariableop6savev2_batch_normalization_30_beta_read_readvariableop=savev2_batch_normalization_30_moving_mean_read_readvariableopAsavev2_batch_normalization_30_moving_variance_read_readvariableop5savev2_conv2d_transpose_18_kernel_read_readvariableop3savev2_conv2d_transpose_18_bias_read_readvariableop7savev2_batch_normalization_31_gamma_read_readvariableop6savev2_batch_normalization_31_beta_read_readvariableop=savev2_batch_normalization_31_moving_mean_read_readvariableopAsavev2_batch_normalization_31_moving_variance_read_readvariableop5savev2_conv2d_transpose_19_kernel_read_readvariableop3savev2_conv2d_transpose_19_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : 0:0:0:0:0:0:0@:@:@:@:@:@:@:::::::::::: : : : : : :0 :0:0:0:0:0:@0:@:@:@:@:@:@:: : : : : : : : : : : : : : 2(
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
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: 
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326337

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
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325719

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
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326137

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
U0*K
_output_shapes9
7:?????????@@ : : : : :*
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
:?????????@@ 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5321695

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
?
?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325431

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
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5322928

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
F__inference_conv2d_14_layer_call_and_return_conditional_losses_5322805

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
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
:?????????  @2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5321443

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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325919

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
?
?
%__inference_signature_wrapper_5324463
input_4!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35:0 

unknown_36:0

unknown_37:0

unknown_38:0

unknown_39:0

unknown_40:0$

unknown_41:@0

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_53212512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325269

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5321651

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325575

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
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_31_layer_call_fn_5326426

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
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_53225072
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
?
?
5__inference_conv2d_transpose_16_layer_call_fn_5325955

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_53229612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5323547

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
7:?????????  @:@:@:@:@:*
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
:?????????  @2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
F__inference_conv2d_12_layer_call_and_return_conditional_losses_5322717

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
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
:??????????? 2	
BiasAddb
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_30_layer_call_fn_5326265

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_53233342
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
?
?
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5326213

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0
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
:???????????0*
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
:???????????02	
BiasAddb
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
??
?4
D__inference_model_3_layer_call_and_return_conditional_losses_5325161

inputsB
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: <
.batch_normalization_24_readvariableop_resource: >
0batch_normalization_24_readvariableop_1_resource: M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_13_conv2d_readvariableop_resource: 07
)conv2d_13_biasadd_readvariableop_resource:0<
.batch_normalization_25_readvariableop_resource:0>
0batch_normalization_25_readvariableop_1_resource:0M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource:0O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:0B
(conv2d_14_conv2d_readvariableop_resource:0@7
)conv2d_14_biasadd_readvariableop_resource:@<
.batch_normalization_26_readvariableop_resource:@>
0batch_normalization_26_readvariableop_1_resource:@M
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_15_conv2d_readvariableop_resource:@7
)conv2d_15_biasadd_readvariableop_resource:<
.batch_normalization_27_readvariableop_resource:>
0batch_normalization_27_readvariableop_1_resource:M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_15_biasadd_readvariableop_resource:<
.batch_normalization_28_readvariableop_resource:>
0batch_normalization_28_readvariableop_1_resource:M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:V
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_16_biasadd_readvariableop_resource: <
.batch_normalization_29_readvariableop_resource: >
0batch_normalization_29_readvariableop_1_resource: M
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource:0 A
3conv2d_transpose_17_biasadd_readvariableop_resource:0<
.batch_normalization_30_readvariableop_resource:0>
0batch_normalization_30_readvariableop_1_resource:0M
?batch_normalization_30_fusedbatchnormv3_readvariableop_resource:0O
Abatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:0V
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource:@0A
3conv2d_transpose_18_biasadd_readvariableop_resource:@<
.batch_normalization_31_readvariableop_resource:@>
0batch_normalization_31_readvariableop_1_resource:@M
?batch_normalization_31_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_19_biasadd_readvariableop_resource:
identity??%batch_normalization_24/AssignNewValue?'batch_normalization_24/AssignNewValue_1?6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_24/ReadVariableOp?'batch_normalization_24/ReadVariableOp_1?%batch_normalization_25/AssignNewValue?'batch_normalization_25/AssignNewValue_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1?%batch_normalization_26/AssignNewValue?'batch_normalization_26/AssignNewValue_1?6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?%batch_normalization_29/AssignNewValue?'batch_normalization_29/AssignNewValue_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1?%batch_normalization_30/AssignNewValue?'batch_normalization_30/AssignNewValue_1?6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_30/ReadVariableOp?'batch_normalization_30/ReadVariableOp_1?%batch_normalization_31/AssignNewValue?'batch_normalization_31/AssignNewValue_1?6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_31/ReadVariableOp?'batch_normalization_31/ReadVariableOp_1? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?*conv2d_transpose_16/BiasAdd/ReadVariableOp?3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?*conv2d_transpose_17/BiasAdd/ReadVariableOp?3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_12/BiasAdd?
conv2d_12/TanhTanhconv2d_12/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_12/Tanh?
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_24/ReadVariableOp?
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_24/ReadVariableOp_1?
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_12/Tanh:y:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_24/FusedBatchNormV3?
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_24/AssignNewValue?
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_24/AssignNewValue_1?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@0*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@02
conv2d_13/BiasAdd~
conv2d_13/TanhTanhconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@02
conv2d_13/Tanh?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:0*
dtype02'
%batch_normalization_25/ReadVariableOp?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:0*
dtype02)
'batch_normalization_25/ReadVariableOp_1?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_13/Tanh:y:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_25/FusedBatchNormV3?
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_25/AssignNewValue?
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_25/AssignNewValue_1?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_14/BiasAdd~
conv2d_14/TanhTanhconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_14/Tanh?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3conv2d_14/Tanh:y:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_26/FusedBatchNormV3?
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue?
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_15/BiasAdd~
conv2d_15/TanhTanhconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_15/Tanh?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_15/Tanh:y:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_27/FusedBatchNormV3?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1?
conv2d_transpose_15/ShapeShape+batch_normalization_27/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape?
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack?
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1?
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_15/stack/2|
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/3?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack?
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stack?
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1?
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transpose?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_15/BiasAdd/ReadVariableOp?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_15/BiasAdd?
conv2d_transpose_15/TanhTanh$conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_15/Tanh?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_15/Tanh:y:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_28/FusedBatchNormV3?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_28/AssignNewValue?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_28/AssignNewValue_1?
conv2d_transpose_16/ShapeShape+batch_normalization_28/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape?
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack?
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1?
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2?
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_16/stack/2|
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_16/stack/3?
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack?
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stack?
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1?
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2?
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1?
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transpose?
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOp?
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_16/BiasAdd?
conv2d_transpose_16/TanhTanh$conv2d_transpose_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_16/Tanh?
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_29/ReadVariableOp?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_29/ReadVariableOp_1?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_16/Tanh:y:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_29/FusedBatchNormV3?
%batch_normalization_29/AssignNewValueAssignVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource4batch_normalization_29/FusedBatchNormV3:batch_mean:07^batch_normalization_29/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_29/AssignNewValue?
'batch_normalization_29/AssignNewValue_1AssignVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_29/FusedBatchNormV3:batch_variance:09^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_29/AssignNewValue_1?
conv2d_transpose_17/ShapeShape+batch_normalization_29/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape?
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack?
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1?
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2?
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice}
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_17/stack/1}
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_17/stack/2|
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_17/stack/3?
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack?
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stack?
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1?
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2?
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1?
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0 *
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_29/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transpose?
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOp?
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_transpose_17/BiasAdd?
conv2d_transpose_17/TanhTanh$conv2d_transpose_17/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_transpose_17/Tanh?
%batch_normalization_30/ReadVariableOpReadVariableOp.batch_normalization_30_readvariableop_resource*
_output_shapes
:0*
dtype02'
%batch_normalization_30/ReadVariableOp?
'batch_normalization_30/ReadVariableOp_1ReadVariableOp0batch_normalization_30_readvariableop_1_resource*
_output_shapes
:0*
dtype02)
'batch_normalization_30/ReadVariableOp_1?
6batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype028
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02:
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_30/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_17/Tanh:y:0-batch_normalization_30/ReadVariableOp:value:0/batch_normalization_30/ReadVariableOp_1:value:0>batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_30/FusedBatchNormV3?
%batch_normalization_30/AssignNewValueAssignVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource4batch_normalization_30/FusedBatchNormV3:batch_mean:07^batch_normalization_30/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_30/AssignNewValue?
'batch_normalization_30/AssignNewValue_1AssignVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_30/FusedBatchNormV3:batch_variance:09^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_30/AssignNewValue_1?
conv2d_transpose_18/ShapeShape+batch_normalization_30/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_18/Shape?
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_18/strided_slice/stack?
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_1?
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_2?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_18/strided_slice}
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_18/stack/1}
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_18/stack/2|
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_18/stack/3?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_18/stack?
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_18/strided_slice_1/stack?
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_1?
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_2?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_18/strided_slice_1?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@0*
dtype025
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_30/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2&
$conv2d_transpose_18/conv2d_transpose?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_18/BiasAdd/ReadVariableOp?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_18/BiasAdd?
conv2d_transpose_18/TanhTanh$conv2d_transpose_18/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_18/Tanh?
%batch_normalization_31/ReadVariableOpReadVariableOp.batch_normalization_31_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_31/ReadVariableOp?
'batch_normalization_31/ReadVariableOp_1ReadVariableOp0batch_normalization_31_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_31/ReadVariableOp_1?
6batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_31/FusedBatchNormV3FusedBatchNormV3conv2d_transpose_18/Tanh:y:0-batch_normalization_31/ReadVariableOp:value:0/batch_normalization_31/ReadVariableOp_1:value:0>batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_31/FusedBatchNormV3?
%batch_normalization_31/AssignNewValueAssignVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource4batch_normalization_31/FusedBatchNormV3:batch_mean:07^batch_normalization_31/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_31/AssignNewValue?
'batch_normalization_31/AssignNewValue_1AssignVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_31/FusedBatchNormV3:batch_variance:09^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_31/AssignNewValue_1?
conv2d_transpose_19/ShapeShape+batch_normalization_31/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_19/Shape?
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_19/strided_slice/stack?
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_1?
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_2?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_19/strided_slice}
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_19/stack/1}
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_19/stack/2|
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_19/stack/3?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_19/stack?
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_19/strided_slice_1/stack?
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_1?
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_2?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_19/strided_slice_1?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_31/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$conv2d_transpose_19/conv2d_transpose?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_19/BiasAdd/ReadVariableOp?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_19/BiasAdd?
conv2d_transpose_19/SigmoidSigmoid$conv2d_transpose_19/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_19/Sigmoid?
IdentityIdentityconv2d_transpose_19/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1&^batch_normalization_29/AssignNewValue(^batch_normalization_29/AssignNewValue_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_1&^batch_normalization_30/AssignNewValue(^batch_normalization_30/AssignNewValue_17^batch_normalization_30/FusedBatchNormV3/ReadVariableOp9^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_30/ReadVariableOp(^batch_normalization_30/ReadVariableOp_1&^batch_normalization_31/AssignNewValue(^batch_normalization_31/AssignNewValue_17^batch_normalization_31/FusedBatchNormV3/ReadVariableOp9^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_31/ReadVariableOp(^batch_normalization_31/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12N
%batch_normalization_29/AssignNewValue%batch_normalization_29/AssignNewValue2R
'batch_normalization_29/AssignNewValue_1'batch_normalization_29/AssignNewValue_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12N
%batch_normalization_30/AssignNewValue%batch_normalization_30/AssignNewValue2R
'batch_normalization_30/AssignNewValue_1'batch_normalization_30/AssignNewValue_12p
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp6batch_normalization_30/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_18batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_30/ReadVariableOp%batch_normalization_30/ReadVariableOp2R
'batch_normalization_30/ReadVariableOp_1'batch_normalization_30/ReadVariableOp_12N
%batch_normalization_31/AssignNewValue%batch_normalization_31/AssignNewValue2R
'batch_normalization_31/AssignNewValue_1'batch_normalization_31/AssignNewValue_12p
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp6batch_normalization_31/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_18batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_31/ReadVariableOp%batch_normalization_31/ReadVariableOp2R
'batch_normalization_31/ReadVariableOp_1'batch_normalization_31/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_31_layer_call_fn_5326465

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
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_53232812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5321399

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
?
?
F__inference_conv2d_13_layer_call_and_return_conditional_losses_5322761

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@0*
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
T0*/
_output_shapes
:?????????@@02	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@02
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
F__inference_conv2d_14_layer_call_and_return_conditional_losses_5325469

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
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
:?????????  @2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325305

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
?
?
+__inference_conv2d_12_layer_call_fn_5325170

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_53227172
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
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?%
?
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5322007

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5321909

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
?
?
8__inference_batch_normalization_25_layer_call_fn_5325377

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
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_53236012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325937

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
?%
?
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5325989

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_30_layer_call_fn_5326226

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
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_53222932
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
?
?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5322849

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326537

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
9:???????????@:@:@:@:@:*
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
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_24_layer_call_fn_5325207

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_53213172
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5322872

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
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_24_layer_call_fn_5325233

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_53236552
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
?
?
8__inference_batch_normalization_29_layer_call_fn_5326052

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
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_53229842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_15_layer_call_fn_5325746

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_53217932
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
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5323281

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
9:???????????@:@:@:@:@:*
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
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325413

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
?%
?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5321793

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_28_layer_call_fn_5325826

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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_53218652
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
?w
?
D__inference_model_3_layer_call_and_return_conditional_losses_5323136

inputs+
conv2d_12_5322718: 
conv2d_12_5322720: ,
batch_normalization_24_5322741: ,
batch_normalization_24_5322743: ,
batch_normalization_24_5322745: ,
batch_normalization_24_5322747: +
conv2d_13_5322762: 0
conv2d_13_5322764:0,
batch_normalization_25_5322785:0,
batch_normalization_25_5322787:0,
batch_normalization_25_5322789:0,
batch_normalization_25_5322791:0+
conv2d_14_5322806:0@
conv2d_14_5322808:@,
batch_normalization_26_5322829:@,
batch_normalization_26_5322831:@,
batch_normalization_26_5322833:@,
batch_normalization_26_5322835:@+
conv2d_15_5322850:@
conv2d_15_5322852:,
batch_normalization_27_5322873:,
batch_normalization_27_5322875:,
batch_normalization_27_5322877:,
batch_normalization_27_5322879:5
conv2d_transpose_15_5322906:)
conv2d_transpose_15_5322908:,
batch_normalization_28_5322929:,
batch_normalization_28_5322931:,
batch_normalization_28_5322933:,
batch_normalization_28_5322935:5
conv2d_transpose_16_5322962: )
conv2d_transpose_16_5322964: ,
batch_normalization_29_5322985: ,
batch_normalization_29_5322987: ,
batch_normalization_29_5322989: ,
batch_normalization_29_5322991: 5
conv2d_transpose_17_5323018:0 )
conv2d_transpose_17_5323020:0,
batch_normalization_30_5323041:0,
batch_normalization_30_5323043:0,
batch_normalization_30_5323045:0,
batch_normalization_30_5323047:05
conv2d_transpose_18_5323074:@0)
conv2d_transpose_18_5323076:@,
batch_normalization_31_5323097:@,
batch_normalization_31_5323099:@,
batch_normalization_31_5323101:@,
batch_normalization_31_5323103:@5
conv2d_transpose_19_5323130:@)
conv2d_transpose_19_5323132:
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?+conv2d_transpose_16/StatefulPartitionedCall?+conv2d_transpose_17/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_5322718conv2d_12_5322720*
Tin
2*
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
GPU2*0J 8? *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_53227172#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_5322741batch_normalization_24_5322743batch_normalization_24_5322745batch_normalization_24_5322747*
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_532274020
.batch_normalization_24/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_13_5322762conv2d_13_5322764*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_53227612#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_5322785batch_normalization_25_5322787batch_normalization_25_5322789batch_normalization_25_5322791*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_532278420
.batch_normalization_25/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_14_5322806conv2d_14_5322808*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_53228052#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_26_5322829batch_normalization_26_5322831batch_normalization_26_5322833batch_normalization_26_5322835*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_532282820
.batch_normalization_26/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_15_5322850conv2d_15_5322852*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_53228492#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_27_5322873batch_normalization_27_5322875batch_normalization_27_5322877batch_normalization_27_5322879*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_532287220
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_15_5322906conv2d_transpose_15_5322908*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_53229052-
+conv2d_transpose_15/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0batch_normalization_28_5322929batch_normalization_28_5322931batch_normalization_28_5322933batch_normalization_28_5322935*
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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_532292820
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_16_5322962conv2d_transpose_16_5322964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_53229612-
+conv2d_transpose_16/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0batch_normalization_29_5322985batch_normalization_29_5322987batch_normalization_29_5322989batch_normalization_29_5322991*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_532298420
.batch_normalization_29/StatefulPartitionedCall?
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv2d_transpose_17_5323018conv2d_transpose_17_5323020*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_53230172-
+conv2d_transpose_17/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0batch_normalization_30_5323041batch_normalization_30_5323043batch_normalization_30_5323045batch_normalization_30_5323047*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_532304020
.batch_normalization_30/StatefulPartitionedCall?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv2d_transpose_18_5323074conv2d_transpose_18_5323076*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_53230732-
+conv2d_transpose_18/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0batch_normalization_31_5323097batch_normalization_31_5323099batch_normalization_31_5323101batch_normalization_31_5323103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_532309620
.batch_normalization_31/StatefulPartitionedCall?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_transpose_19_5323130conv2d_transpose_19_5323132*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_53231292-
+conv2d_transpose_19/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_19/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5325613

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5321865

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
8__inference_batch_normalization_28_layer_call_fn_5325865

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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_53234402
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
?
?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5325813

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
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
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
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
:?????????  *
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
:?????????  2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?(
#__inference__traced_restore_5327024
file_prefix;
!assignvariableop_conv2d_12_kernel: /
!assignvariableop_1_conv2d_12_bias: =
/assignvariableop_2_batch_normalization_24_gamma: <
.assignvariableop_3_batch_normalization_24_beta: C
5assignvariableop_4_batch_normalization_24_moving_mean: G
9assignvariableop_5_batch_normalization_24_moving_variance: =
#assignvariableop_6_conv2d_13_kernel: 0/
!assignvariableop_7_conv2d_13_bias:0=
/assignvariableop_8_batch_normalization_25_gamma:0<
.assignvariableop_9_batch_normalization_25_beta:0D
6assignvariableop_10_batch_normalization_25_moving_mean:0H
:assignvariableop_11_batch_normalization_25_moving_variance:0>
$assignvariableop_12_conv2d_14_kernel:0@0
"assignvariableop_13_conv2d_14_bias:@>
0assignvariableop_14_batch_normalization_26_gamma:@=
/assignvariableop_15_batch_normalization_26_beta:@D
6assignvariableop_16_batch_normalization_26_moving_mean:@H
:assignvariableop_17_batch_normalization_26_moving_variance:@>
$assignvariableop_18_conv2d_15_kernel:@0
"assignvariableop_19_conv2d_15_bias:>
0assignvariableop_20_batch_normalization_27_gamma:=
/assignvariableop_21_batch_normalization_27_beta:D
6assignvariableop_22_batch_normalization_27_moving_mean:H
:assignvariableop_23_batch_normalization_27_moving_variance:H
.assignvariableop_24_conv2d_transpose_15_kernel::
,assignvariableop_25_conv2d_transpose_15_bias:>
0assignvariableop_26_batch_normalization_28_gamma:=
/assignvariableop_27_batch_normalization_28_beta:D
6assignvariableop_28_batch_normalization_28_moving_mean:H
:assignvariableop_29_batch_normalization_28_moving_variance:H
.assignvariableop_30_conv2d_transpose_16_kernel: :
,assignvariableop_31_conv2d_transpose_16_bias: >
0assignvariableop_32_batch_normalization_29_gamma: =
/assignvariableop_33_batch_normalization_29_beta: D
6assignvariableop_34_batch_normalization_29_moving_mean: H
:assignvariableop_35_batch_normalization_29_moving_variance: H
.assignvariableop_36_conv2d_transpose_17_kernel:0 :
,assignvariableop_37_conv2d_transpose_17_bias:0>
0assignvariableop_38_batch_normalization_30_gamma:0=
/assignvariableop_39_batch_normalization_30_beta:0D
6assignvariableop_40_batch_normalization_30_moving_mean:0H
:assignvariableop_41_batch_normalization_30_moving_variance:0H
.assignvariableop_42_conv2d_transpose_18_kernel:@0:
,assignvariableop_43_conv2d_transpose_18_bias:@>
0assignvariableop_44_batch_normalization_31_gamma:@=
/assignvariableop_45_batch_normalization_31_beta:@D
6assignvariableop_46_batch_normalization_31_moving_mean:@H
:assignvariableop_47_batch_normalization_31_moving_variance:@H
.assignvariableop_48_conv2d_transpose_19_kernel:@:
,assignvariableop_49_conv2d_transpose_19_bias:&
assignvariableop_50_sgd_iter:	 '
assignvariableop_51_sgd_decay: *
 assignvariableop_52_sgd_momentum: #
assignvariableop_53_total: #
assignvariableop_54_count: %
assignvariableop_55_total_1: %
assignvariableop_56_count_1: %
assignvariableop_57_total_2: %
assignvariableop_58_count_2: %
assignvariableop_59_total_3: %
assignvariableop_60_count_3: %
assignvariableop_61_total_4: %
assignvariableop_62_count_4: 
identity_64??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_24_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_24_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_24_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_24_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_25_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_25_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_25_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_25_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_26_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_26_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_26_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_26_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_27_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_27_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_27_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_27_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_conv2d_transpose_15_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_conv2d_transpose_15_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_28_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_28_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_28_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_28_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp.assignvariableop_30_conv2d_transpose_16_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp,assignvariableop_31_conv2d_transpose_16_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_29_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_29_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_29_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_29_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp.assignvariableop_36_conv2d_transpose_17_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp,assignvariableop_37_conv2d_transpose_17_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_30_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_30_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_30_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_30_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_conv2d_transpose_18_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_conv2d_transpose_18_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_31_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_31_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_31_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_31_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp.assignvariableop_48_conv2d_transpose_19_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp,assignvariableop_49_conv2d_transpose_19_biasIdentity_49:output:0"/device:CPU:0*
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
AssignVariableOp_53AssignVariableOpassignvariableop_53_totalIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_countIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_total_2Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_count_2Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_3Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_3Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_4Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_4Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63f
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_64?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_64Identity_64:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5326413

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@
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
:???????????@*
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
:???????????@2	
BiasAddb
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5322507

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325251

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
8__inference_batch_normalization_24_layer_call_fn_5325194

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_53212732
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325287

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
?	
?
8__inference_batch_normalization_30_layer_call_fn_5326239

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
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_53223372
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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325395

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
?	
?
8__inference_batch_normalization_26_layer_call_fn_5325495

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_53215692
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
8__inference_batch_normalization_29_layer_call_fn_5326026

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
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_53220792
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
?&
?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5326589

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
?&
?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5322649

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
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326083

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
?
?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5323129

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
:???????????*
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
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoidp
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_27_layer_call_fn_5325652

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
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_53228722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_16_layer_call_fn_5325946

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_53220072
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
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326519

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
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5321525

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
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5323040

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
?
?
8__inference_batch_normalization_26_layer_call_fn_5325521

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
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_53235472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326301

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
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5322551

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
?
?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5322905

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
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
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
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
:?????????  *
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
:?????????  2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Tanhk
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326119

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
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325883

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
?	
?
8__inference_batch_normalization_31_layer_call_fn_5326452

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
:???????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_53230962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?%
?
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5322221

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_3_layer_call_fn_5323239
input_4!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35:0 

unknown_36:0

unknown_37:0

unknown_38:0

unknown_39:0

unknown_40:0$

unknown_41:@0

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_53231362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5323334

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
?w
?
D__inference_model_3_layer_call_and_return_conditional_losses_5323904

inputs+
conv2d_12_5323786: 
conv2d_12_5323788: ,
batch_normalization_24_5323791: ,
batch_normalization_24_5323793: ,
batch_normalization_24_5323795: ,
batch_normalization_24_5323797: +
conv2d_13_5323800: 0
conv2d_13_5323802:0,
batch_normalization_25_5323805:0,
batch_normalization_25_5323807:0,
batch_normalization_25_5323809:0,
batch_normalization_25_5323811:0+
conv2d_14_5323814:0@
conv2d_14_5323816:@,
batch_normalization_26_5323819:@,
batch_normalization_26_5323821:@,
batch_normalization_26_5323823:@,
batch_normalization_26_5323825:@+
conv2d_15_5323828:@
conv2d_15_5323830:,
batch_normalization_27_5323833:,
batch_normalization_27_5323835:,
batch_normalization_27_5323837:,
batch_normalization_27_5323839:5
conv2d_transpose_15_5323842:)
conv2d_transpose_15_5323844:,
batch_normalization_28_5323847:,
batch_normalization_28_5323849:,
batch_normalization_28_5323851:,
batch_normalization_28_5323853:5
conv2d_transpose_16_5323856: )
conv2d_transpose_16_5323858: ,
batch_normalization_29_5323861: ,
batch_normalization_29_5323863: ,
batch_normalization_29_5323865: ,
batch_normalization_29_5323867: 5
conv2d_transpose_17_5323870:0 )
conv2d_transpose_17_5323872:0,
batch_normalization_30_5323875:0,
batch_normalization_30_5323877:0,
batch_normalization_30_5323879:0,
batch_normalization_30_5323881:05
conv2d_transpose_18_5323884:@0)
conv2d_transpose_18_5323886:@,
batch_normalization_31_5323889:@,
batch_normalization_31_5323891:@,
batch_normalization_31_5323893:@,
batch_normalization_31_5323895:@5
conv2d_transpose_19_5323898:@)
conv2d_transpose_19_5323900:
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?+conv2d_transpose_16/StatefulPartitionedCall?+conv2d_transpose_17/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_5323786conv2d_12_5323788*
Tin
2*
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
GPU2*0J 8? *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_53227172#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_5323791batch_normalization_24_5323793batch_normalization_24_5323795batch_normalization_24_5323797*
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_532365520
.batch_normalization_24/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_13_5323800conv2d_13_5323802*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_53227612#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_5323805batch_normalization_25_5323807batch_normalization_25_5323809batch_normalization_25_5323811*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_532360120
.batch_normalization_25/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_14_5323814conv2d_14_5323816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_53228052#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_26_5323819batch_normalization_26_5323821batch_normalization_26_5323823batch_normalization_26_5323825*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_532354720
.batch_normalization_26/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_15_5323828conv2d_15_5323830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_53228492#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_27_5323833batch_normalization_27_5323835batch_normalization_27_5323837batch_normalization_27_5323839*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_532349320
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_15_5323842conv2d_transpose_15_5323844*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_53229052-
+conv2d_transpose_15/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0batch_normalization_28_5323847batch_normalization_28_5323849batch_normalization_28_5323851batch_normalization_28_5323853*
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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_532344020
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_16_5323856conv2d_transpose_16_5323858*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_53229612-
+conv2d_transpose_16/StatefulPartitionedCall?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0batch_normalization_29_5323861batch_normalization_29_5323863batch_normalization_29_5323865batch_normalization_29_5323867*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_532338720
.batch_normalization_29/StatefulPartitionedCall?
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv2d_transpose_17_5323870conv2d_transpose_17_5323872*
Tin
2*
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
GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_53230172-
+conv2d_transpose_17/StatefulPartitionedCall?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0batch_normalization_30_5323875batch_normalization_30_5323877batch_normalization_30_5323879batch_normalization_30_5323881*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_532333420
.batch_normalization_30/StatefulPartitionedCall?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv2d_transpose_18_5323884conv2d_transpose_18_5323886*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_53230732-
+conv2d_transpose_18/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0batch_normalization_31_5323889batch_normalization_31_5323891batch_normalization_31_5323893batch_normalization_31_5323895*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_532328120
.batch_normalization_31/StatefulPartitionedCall?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_transpose_19_5323898conv2d_transpose_19_5323900*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_53231292-
+conv2d_transpose_19/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_19/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325593

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
7:?????????  @:@:@:@:@:*
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
:?????????  @2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?%
?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5325789

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_15_layer_call_fn_5325602

inputs!
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_53228492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5323493

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
7:?????????:::::*
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
:?????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_28_layer_call_fn_5325839

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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_53219092
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
?
?
)__inference_model_3_layer_call_fn_5324112
input_4!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35:0 

unknown_36:0

unknown_37:0

unknown_38:0

unknown_39:0

unknown_40:0$

unknown_41:@0

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_53239042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
)__inference_model_3_layer_call_fn_5324673

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: 0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35:0 

unknown_36:0

unknown_37:0

unknown_38:0

unknown_39:0

unknown_40:0$

unknown_41:@0

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@

unknown_48:
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_53239042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5323017

inputsB
(conv2d_transpose_readvariableop_resource:0 -
biasadd_readvariableop_resource:0
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
:???????????0*
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
:???????????02	
BiasAddb
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_28_layer_call_fn_5325852

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
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_53229282
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
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326101

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
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5322984

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
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@ 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5322740

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
?%
?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5326389

inputsB
(conv2d_transpose_readvariableop_resource:@0-
biasadd_readvariableop_resource:@
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
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_25_layer_call_fn_5325351

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_53214432
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5321273

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
?
?
+__inference_conv2d_13_layer_call_fn_5325314

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_53227612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
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
input_4:
serving_default_input_4:0???????????Q
conv2d_transpose_19:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5	variables
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
9
	?iter

?decay
?momentum"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
812
913
?14
@15
A16
B17
G18
H19
N20
O21
P22
Q23
V24
W25
]26
^27
_28
`29
e30
f31
l32
m33
n34
o35
t36
u37
{38
|39
}40
~41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
0
1
!2
"3
)4
*5
06
17
88
99
?10
@11
G12
H13
N14
O15
V16
W17
]18
^19
e20
f21
l22
m23
t24
u25
{26
|27
?28
?29
?30
?31
?32
?33"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
regularization_losses
?layers
	variables
trainable_variables
?non_trainable_variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
	variables
trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_24/gamma
):' 2batch_normalization_24/beta
2:0  (2"batch_normalization_24/moving_mean
6:4  (2&batch_normalization_24/moving_variance
 "
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
%regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
&	variables
'trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 02conv2d_13/kernel
:02conv2d_13/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
+regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
,	variables
-trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(02batch_normalization_25/gamma
):'02batch_normalization_25/beta
2:00 (2"batch_normalization_25/moving_mean
6:40 (2&batch_normalization_25/moving_variance
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
4regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
5	variables
6trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(0@2conv2d_14/kernel
:@2conv2d_14/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
:regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
;	variables
<trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_26/gamma
):'@2batch_normalization_26/beta
2:0@ (2"batch_normalization_26/moving_mean
6:4@ (2&batch_normalization_26/moving_variance
 "
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
Cregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
D	variables
Etrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_15/kernel
:2conv2d_15/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
Iregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
J	variables
Ktrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_27/gamma
):'2batch_normalization_27/beta
2:0 (2"batch_normalization_27/moving_mean
6:4 (2&batch_normalization_27/moving_variance
 "
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
Rregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
S	variables
Ttrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:22conv2d_transpose_15/kernel
&:$2conv2d_transpose_15/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
Xregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
Y	variables
Ztrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_28/gamma
):'2batch_normalization_28/beta
2:0 (2"batch_normalization_28/moving_mean
6:4 (2&batch_normalization_28/moving_variance
 "
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
aregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
b	variables
ctrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2 2conv2d_transpose_16/kernel
&:$ 2conv2d_transpose_16/bias
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
gregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
h	variables
itrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_29/gamma
):' 2batch_normalization_29/beta
2:0  (2"batch_normalization_29/moving_mean
6:4  (2&batch_normalization_29/moving_variance
 "
trackable_list_wrapper
<
l0
m1
n2
o3"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
pregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
q	variables
rtrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:20 2conv2d_transpose_17/kernel
&:$02conv2d_transpose_17/bias
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
?
vregularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
w	variables
xtrainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(02batch_normalization_30/gamma
):'02batch_normalization_30/beta
2:00 (2"batch_normalization_30/moving_mean
6:40 (2&batch_normalization_30/moving_variance
 "
trackable_list_wrapper
<
{0
|1
}2
~3"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2@02conv2d_transpose_18/kernel
&:$@2conv2d_transpose_18/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_31/gamma
):'@2batch_normalization_31/beta
2:0@ (2"batch_normalization_31/moving_mean
6:4@ (2&batch_normalization_31/moving_variance
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2@2conv2d_transpose_19/kernel
&:$2conv2d_transpose_19/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?trainable_variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/momentum
 "
trackable_dict_wrapper
 "
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
?
#0
$1
22
33
A4
B5
P6
Q7
_8
`9
n10
o11
}12
~13
?14
?15"
trackable_list_wrapper
H
?0
?1
?2
?3
?4"
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
.
#0
$1"
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
.
20
31"
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
.
A0
B1"
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
.
P0
Q1"
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
.
_0
`1"
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
.
n0
o1"
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
.
}0
~1"
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
0
?0
?1"
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

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
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
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
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
?2?
)__inference_model_3_layer_call_fn_5323239
)__inference_model_3_layer_call_fn_5324568
)__inference_model_3_layer_call_fn_5324673
)__inference_model_3_layer_call_fn_5324112?
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
?B?
"__inference__wrapped_model_5321251input_4"?
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
D__inference_model_3_layer_call_and_return_conditional_losses_5324917
D__inference_model_3_layer_call_and_return_conditional_losses_5325161
D__inference_model_3_layer_call_and_return_conditional_losses_5324233
D__inference_model_3_layer_call_and_return_conditional_losses_5324354?
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
+__inference_conv2d_12_layer_call_fn_5325170?
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
F__inference_conv2d_12_layer_call_and_return_conditional_losses_5325181?
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
?2?
8__inference_batch_normalization_24_layer_call_fn_5325194
8__inference_batch_normalization_24_layer_call_fn_5325207
8__inference_batch_normalization_24_layer_call_fn_5325220
8__inference_batch_normalization_24_layer_call_fn_5325233?
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
?2?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325251
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325269
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325287
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325305?
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
+__inference_conv2d_13_layer_call_fn_5325314?
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
F__inference_conv2d_13_layer_call_and_return_conditional_losses_5325325?
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
?2?
8__inference_batch_normalization_25_layer_call_fn_5325338
8__inference_batch_normalization_25_layer_call_fn_5325351
8__inference_batch_normalization_25_layer_call_fn_5325364
8__inference_batch_normalization_25_layer_call_fn_5325377?
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
?2?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325395
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325413
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325431
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325449?
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
+__inference_conv2d_14_layer_call_fn_5325458?
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
F__inference_conv2d_14_layer_call_and_return_conditional_losses_5325469?
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
?2?
8__inference_batch_normalization_26_layer_call_fn_5325482
8__inference_batch_normalization_26_layer_call_fn_5325495
8__inference_batch_normalization_26_layer_call_fn_5325508
8__inference_batch_normalization_26_layer_call_fn_5325521?
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
?2?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325539
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325557
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325575
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325593?
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
+__inference_conv2d_15_layer_call_fn_5325602?
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5325613?
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
?2?
8__inference_batch_normalization_27_layer_call_fn_5325626
8__inference_batch_normalization_27_layer_call_fn_5325639
8__inference_batch_normalization_27_layer_call_fn_5325652
8__inference_batch_normalization_27_layer_call_fn_5325665?
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
?2?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325683
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325701
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325719
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325737?
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
5__inference_conv2d_transpose_15_layer_call_fn_5325746
5__inference_conv2d_transpose_15_layer_call_fn_5325755?
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
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5325789
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5325813?
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
?2?
8__inference_batch_normalization_28_layer_call_fn_5325826
8__inference_batch_normalization_28_layer_call_fn_5325839
8__inference_batch_normalization_28_layer_call_fn_5325852
8__inference_batch_normalization_28_layer_call_fn_5325865?
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
?2?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325883
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325901
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325919
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325937?
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
5__inference_conv2d_transpose_16_layer_call_fn_5325946
5__inference_conv2d_transpose_16_layer_call_fn_5325955?
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
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5325989
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5326013?
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
?2?
8__inference_batch_normalization_29_layer_call_fn_5326026
8__inference_batch_normalization_29_layer_call_fn_5326039
8__inference_batch_normalization_29_layer_call_fn_5326052
8__inference_batch_normalization_29_layer_call_fn_5326065?
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
?2?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326083
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326101
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326119
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326137?
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
5__inference_conv2d_transpose_17_layer_call_fn_5326146
5__inference_conv2d_transpose_17_layer_call_fn_5326155?
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
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5326189
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5326213?
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
?2?
8__inference_batch_normalization_30_layer_call_fn_5326226
8__inference_batch_normalization_30_layer_call_fn_5326239
8__inference_batch_normalization_30_layer_call_fn_5326252
8__inference_batch_normalization_30_layer_call_fn_5326265?
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
?2?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326283
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326301
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326319
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326337?
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
5__inference_conv2d_transpose_18_layer_call_fn_5326346
5__inference_conv2d_transpose_18_layer_call_fn_5326355?
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
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5326389
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5326413?
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
?2?
8__inference_batch_normalization_31_layer_call_fn_5326426
8__inference_batch_normalization_31_layer_call_fn_5326439
8__inference_batch_normalization_31_layer_call_fn_5326452
8__inference_batch_normalization_31_layer_call_fn_5326465?
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
?2?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326483
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326501
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326519
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326537?
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
5__inference_conv2d_transpose_19_layer_call_fn_5326546
5__inference_conv2d_transpose_19_layer_call_fn_5326555?
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
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5326589
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5326613?
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
%__inference_signature_wrapper_5324463input_4"?
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
 ?
"__inference__wrapped_model_5321251?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????:?7
0?-
+?(
input_4???????????
? "S?P
N
conv2d_transpose_197?4
conv2d_transpose_19????????????
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325251?!"#$M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325269?!"#$M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325287v!"#$=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5325305v!"#$=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
8__inference_batch_normalization_24_layer_call_fn_5325194?!"#$M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_24_layer_call_fn_5325207?!"#$M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_24_layer_call_fn_5325220i!"#$=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
8__inference_batch_normalization_24_layer_call_fn_5325233i!"#$=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325395?0123M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325413?0123M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325431r0123;?8
1?.
(?%
inputs?????????@@0
p 
? "-?*
#? 
0?????????@@0
? ?
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5325449r0123;?8
1?.
(?%
inputs?????????@@0
p
? "-?*
#? 
0?????????@@0
? ?
8__inference_batch_normalization_25_layer_call_fn_5325338?0123M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
8__inference_batch_normalization_25_layer_call_fn_5325351?0123M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
8__inference_batch_normalization_25_layer_call_fn_5325364e0123;?8
1?.
(?%
inputs?????????@@0
p 
? " ??????????@@0?
8__inference_batch_normalization_25_layer_call_fn_5325377e0123;?8
1?.
(?%
inputs?????????@@0
p
? " ??????????@@0?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325539??@ABM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325557??@ABM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325575r?@AB;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5325593r?@AB;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
8__inference_batch_normalization_26_layer_call_fn_5325482??@ABM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_26_layer_call_fn_5325495??@ABM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_26_layer_call_fn_5325508e?@AB;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
8__inference_batch_normalization_26_layer_call_fn_5325521e?@AB;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325683?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325701?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325719rNOPQ;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_5325737rNOPQ;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
8__inference_batch_normalization_27_layer_call_fn_5325626?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_27_layer_call_fn_5325639?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_27_layer_call_fn_5325652eNOPQ;?8
1?.
(?%
inputs?????????
p 
? " ???????????
8__inference_batch_normalization_27_layer_call_fn_5325665eNOPQ;?8
1?.
(?%
inputs?????????
p
? " ???????????
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325883?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325901?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325919r]^_`;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_5325937r]^_`;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
8__inference_batch_normalization_28_layer_call_fn_5325826?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_28_layer_call_fn_5325839?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_28_layer_call_fn_5325852e]^_`;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_28_layer_call_fn_5325865e]^_`;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326083?lmnoM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326101?lmnoM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326119rlmno;?8
1?.
(?%
inputs?????????@@ 
p 
? "-?*
#? 
0?????????@@ 
? ?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_5326137rlmno;?8
1?.
(?%
inputs?????????@@ 
p
? "-?*
#? 
0?????????@@ 
? ?
8__inference_batch_normalization_29_layer_call_fn_5326026?lmnoM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_29_layer_call_fn_5326039?lmnoM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_29_layer_call_fn_5326052elmno;?8
1?.
(?%
inputs?????????@@ 
p 
? " ??????????@@ ?
8__inference_batch_normalization_29_layer_call_fn_5326065elmno;?8
1?.
(?%
inputs?????????@@ 
p
? " ??????????@@ ?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326283?{|}~M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326301?{|}~M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326319v{|}~=?:
3?0
*?'
inputs???????????0
p 
? "/?,
%?"
0???????????0
? ?
S__inference_batch_normalization_30_layer_call_and_return_conditional_losses_5326337v{|}~=?:
3?0
*?'
inputs???????????0
p
? "/?,
%?"
0???????????0
? ?
8__inference_batch_normalization_30_layer_call_fn_5326226?{|}~M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
8__inference_batch_normalization_30_layer_call_fn_5326239?{|}~M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
8__inference_batch_normalization_30_layer_call_fn_5326252i{|}~=?:
3?0
*?'
inputs???????????0
p 
? ""????????????0?
8__inference_batch_normalization_30_layer_call_fn_5326265i{|}~=?:
3?0
*?'
inputs???????????0
p
? ""????????????0?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326483?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326501?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326519z????=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5326537z????=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
8__inference_batch_normalization_31_layer_call_fn_5326426?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_31_layer_call_fn_5326439?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_31_layer_call_fn_5326452m????=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
8__inference_batch_normalization_31_layer_call_fn_5326465m????=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
F__inference_conv2d_12_layer_call_and_return_conditional_losses_5325181p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
+__inference_conv2d_12_layer_call_fn_5325170c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
F__inference_conv2d_13_layer_call_and_return_conditional_losses_5325325n)*9?6
/?,
*?'
inputs??????????? 
? "-?*
#? 
0?????????@@0
? ?
+__inference_conv2d_13_layer_call_fn_5325314a)*9?6
/?,
*?'
inputs??????????? 
? " ??????????@@0?
F__inference_conv2d_14_layer_call_and_return_conditional_losses_5325469l897?4
-?*
(?%
inputs?????????@@0
? "-?*
#? 
0?????????  @
? ?
+__inference_conv2d_14_layer_call_fn_5325458_897?4
-?*
(?%
inputs?????????@@0
? " ??????????  @?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5325613lGH7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_15_layer_call_fn_5325602_GH7?4
-?*
(?%
inputs?????????  @
? " ???????????
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5325789?VWI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_5325813lVW7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????  
? ?
5__inference_conv2d_transpose_15_layer_call_fn_5325746?VWI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_15_layer_call_fn_5325755_VW7?4
-?*
(?%
inputs?????????
? " ??????????  ?
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5325989?efI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_5326013lef7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????@@ 
? ?
5__inference_conv2d_transpose_16_layer_call_fn_5325946?efI?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
5__inference_conv2d_transpose_16_layer_call_fn_5325955_ef7?4
-?*
(?%
inputs?????????  
? " ??????????@@ ?
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5326189?tuI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????0
? ?
P__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_5326213ntu7?4
-?*
(?%
inputs?????????@@ 
? "/?,
%?"
0???????????0
? ?
5__inference_conv2d_transpose_17_layer_call_fn_5326146?tuI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????0?
5__inference_conv2d_transpose_17_layer_call_fn_5326155atu7?4
-?*
(?%
inputs?????????@@ 
? ""????????????0?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5326389???I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_5326413r??9?6
/?,
*?'
inputs???????????0
? "/?,
%?"
0???????????@
? ?
5__inference_conv2d_transpose_18_layer_call_fn_5326346???I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+???????????????????????????@?
5__inference_conv2d_transpose_18_layer_call_fn_5326355e??9?6
/?,
*?'
inputs???????????0
? ""????????????@?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5326589???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_5326613r??9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????
? ?
5__inference_conv2d_transpose_19_layer_call_fn_5326546???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
5__inference_conv2d_transpose_19_layer_call_fn_5326555e??9?6
/?,
*?'
inputs???????????@
? ""?????????????
D__inference_model_3_layer_call_and_return_conditional_losses_5324233?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????B??
8?5
+?(
input_4???????????
p 

 
? "/?,
%?"
0???????????
? ?
D__inference_model_3_layer_call_and_return_conditional_losses_5324354?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????B??
8?5
+?(
input_4???????????
p

 
? "/?,
%?"
0???????????
? ?
D__inference_model_3_layer_call_and_return_conditional_losses_5324917?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
D__inference_model_3_layer_call_and_return_conditional_losses_5325161?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
)__inference_model_3_layer_call_fn_5323239?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????B??
8?5
+?(
input_4???????????
p 

 
? ""?????????????
)__inference_model_3_layer_call_fn_5324112?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????B??
8?5
+?(
input_4???????????
p

 
? ""?????????????
)__inference_model_3_layer_call_fn_5324568?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
)__inference_model_3_layer_call_fn_5324673?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
%__inference_signature_wrapper_5324463?:!"#$)*012389?@ABGHNOPQVW]^_`eflmnotu{|}~????????E?B
? 
;?8
6
input_4+?(
input_4???????????"S?P
N
conv2d_transpose_197?4
conv2d_transpose_19???????????