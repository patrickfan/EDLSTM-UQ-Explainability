��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
h
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
 uq__net_std_tf2_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*1
shared_name" uq__net_std_tf2_1/dense_6/kernel
�
4uq__net_std_tf2_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp uq__net_std_tf2_1/dense_6/kernel*
_output_shapes

:22*
dtype0
�
uq__net_std_tf2_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*/
shared_name uq__net_std_tf2_1/dense_6/bias
�
2uq__net_std_tf2_1/dense_6/bias/Read/ReadVariableOpReadVariableOpuq__net_std_tf2_1/dense_6/bias*
_output_shapes
:2*
dtype0
�
 uq__net_std_tf2_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*1
shared_name" uq__net_std_tf2_1/dense_8/kernel
�
4uq__net_std_tf2_1/dense_8/kernel/Read/ReadVariableOpReadVariableOp uq__net_std_tf2_1/dense_8/kernel*
_output_shapes

:
*
dtype0
�
uq__net_std_tf2_1/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name uq__net_std_tf2_1/dense_8/bias
�
2uq__net_std_tf2_1/dense_8/bias/Read/ReadVariableOpReadVariableOpuq__net_std_tf2_1/dense_8/bias*
_output_shapes
:*
dtype0
�
 uq__net_std_tf2_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*1
shared_name" uq__net_std_tf2_1/dense_7/kernel
�
4uq__net_std_tf2_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp uq__net_std_tf2_1/dense_7/kernel*
_output_shapes

:2
*
dtype0
�
uq__net_std_tf2_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name uq__net_std_tf2_1/dense_7/bias
�
2uq__net_std_tf2_1/dense_7/bias/Read/ReadVariableOpReadVariableOpuq__net_std_tf2_1/dense_7/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
configs
num_nodes_list

inputLayer
fcs
outputLayer
custom_bias
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�
Max_iter
stop_losses

optimizers
lr
test_biases_list
num_neurons_mean
num_neurons_up
num_neurons_down
ypower_root_trans
plot_PI3NN_ylims* 
* 
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*

!0*
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
HB
VARIABLE_VALUEVariable&custom_bias/.ATTRIBUTES/VARIABLE_VALUE*
5
0
1
*2
+3
"4
#5
6*
5
0
1
*2
+3
"4
#5
6*
	
,0* 
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

2serving_default* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

30
41* 
f`
VARIABLE_VALUE uq__net_std_tf2_1/dense_6/kernel,inputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEuq__net_std_tf2_1/dense_6/bias*inputLayer/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
�

*kernel
+bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
ga
VARIABLE_VALUE uq__net_std_tf2_1/dense_8/kernel-outputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEuq__net_std_tf2_1/dense_8/bias+outputLayer/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUE uq__net_std_tf2_1/dense_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEuq__net_std_tf2_1/dense_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
!1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*

*0
+1*
	
,0* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
,0* 
* 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������2*
dtype0*
shape:���������2
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1 uq__net_std_tf2_1/dense_6/kerneluq__net_std_tf2_1/dense_6/bias uq__net_std_tf2_1/dense_7/kerneluq__net_std_tf2_1/dense_7/bias uq__net_std_tf2_1/dense_8/kerneluq__net_std_tf2_1/dense_8/biasVariable*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *-
f(R&
$__inference_signature_wrapper_918532
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp4uq__net_std_tf2_1/dense_6/kernel/Read/ReadVariableOp2uq__net_std_tf2_1/dense_6/bias/Read/ReadVariableOp4uq__net_std_tf2_1/dense_8/kernel/Read/ReadVariableOp2uq__net_std_tf2_1/dense_8/bias/Read/ReadVariableOp4uq__net_std_tf2_1/dense_7/kernel/Read/ReadVariableOp2uq__net_std_tf2_1/dense_7/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *(
f#R!
__inference__traced_save_918684
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable uq__net_std_tf2_1/dense_6/kerneluq__net_std_tf2_1/dense_6/bias uq__net_std_tf2_1/dense_8/kerneluq__net_std_tf2_1/dense_8/bias uq__net_std_tf2_1/dense_7/kerneluq__net_std_tf2_1/dense_7/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *+
f&R$
"__inference__traced_restore_918715��
�/
�
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918432
input_1 
dense_6_918394:22
dense_6_918396:2 
dense_7_918399:2

dense_7_918401:
 
dense_8_918404:

dense_8_918406:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp�Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_918394dense_6_918396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_918248�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_918399dense_7_918401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_918280�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_918404dense_8_918406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_918296r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAdd(dense_8/StatefulPartitionedCall:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T
SquareSquareBiasAdd:output:0*
T0*'
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=Z
addAddV2
Square:y:0add/y:output:0*
T0*'
_output_shapes
:���������G
SqrtSqrtadd:z:0*
T0*'
_output_shapes
:���������w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_7_918399*
_output_shapes

:2
*
dtype0�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/AbsAbsGuq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/SumSum4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/mulMul;uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/x:output:09uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/addAddV2;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const:output:04uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_918399*
_output_shapes

:2
*
dtype0�
3uq__net_std_tf2_1/dense_7/kernel/Regularizer/SquareSquareJuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1Sum7uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1Mul=uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/x:output:0;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1AddV24uq__net_std_tf2_1/dense_7/kernel/Regularizer/add:z:06uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: W
IdentityIdentitySqrt:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall@^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpC^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2�
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp2�
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpBuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:���������2
!
_user_specified_name	input_1
�
�
__inference__traced_save_918684
file_prefix'
#savev2_variable_read_readvariableop?
;savev2_uq__net_std_tf2_1_dense_6_kernel_read_readvariableop=
9savev2_uq__net_std_tf2_1_dense_6_bias_read_readvariableop?
;savev2_uq__net_std_tf2_1_dense_8_kernel_read_readvariableop=
9savev2_uq__net_std_tf2_1_dense_8_bias_read_readvariableop?
;savev2_uq__net_std_tf2_1_dense_7_kernel_read_readvariableop=
9savev2_uq__net_std_tf2_1_dense_7_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&custom_bias/.ATTRIBUTES/VARIABLE_VALUEB,inputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB*inputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB-outputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+outputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop;savev2_uq__net_std_tf2_1_dense_6_kernel_read_readvariableop9savev2_uq__net_std_tf2_1_dense_6_bias_read_readvariableop;savev2_uq__net_std_tf2_1_dense_8_kernel_read_readvariableop9savev2_uq__net_std_tf2_1_dense_8_bias_read_readvariableop;savev2_uq__net_std_tf2_1_dense_7_kernel_read_readvariableop9savev2_uq__net_std_tf2_1_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*M
_input_shapes<
:: ::22:2:
::2
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
::$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:2
: 

_output_shapes
:
:

_output_shapes
: 
�	
�
2__inference_uq__net_std_tf2_1_layer_call_fn_918342
input_1
unknown:22
	unknown_0:2
	unknown_1:2

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *V
fQRO
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������2
!
_user_specified_name	input_1
�"
�
C__inference_dense_7_layer_call_and_return_conditional_losses_918280

inputs0
matmul_readvariableop_resource:2
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp�Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/AbsAbsGuq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/SumSum4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/mulMul;uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/x:output:09uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/addAddV2;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const:output:04uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
3uq__net_std_tf2_1/dense_7/kernel/Regularizer/SquareSquareJuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1Sum7uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1Mul=uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/x:output:0;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1AddV24uq__net_std_tf2_1/dense_7/kernel/Regularizer/add:z:06uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp@^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpC^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp2�
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpBuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_918532
input_1
unknown:22
	unknown_0:2
	unknown_1:2

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� **
f%R#
!__inference__wrapped_model_918231o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������2
!
_user_specified_name	input_1
�	
�
C__inference_dense_6_layer_call_and_return_conditional_losses_918551

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�/
�
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918325
x 
dense_6_918249:22
dense_6_918251:2 
dense_7_918281:2

dense_7_918283:
 
dense_8_918297:

dense_8_918299:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp�Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp�
dense_6/StatefulPartitionedCallStatefulPartitionedCallxdense_6_918249dense_6_918251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_918248�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_918281dense_7_918283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_918280�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_918297dense_8_918299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_918296r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAdd(dense_8/StatefulPartitionedCall:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T
SquareSquareBiasAdd:output:0*
T0*'
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=Z
addAddV2
Square:y:0add/y:output:0*
T0*'
_output_shapes
:���������G
SqrtSqrtadd:z:0*
T0*'
_output_shapes
:���������w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_7_918281*
_output_shapes

:2
*
dtype0�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/AbsAbsGuq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/SumSum4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/mulMul;uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/x:output:09uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/addAddV2;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const:output:04uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_918281*
_output_shapes

:2
*
dtype0�
3uq__net_std_tf2_1/dense_7/kernel/Regularizer/SquareSquareJuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1Sum7uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1Mul=uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/x:output:0;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1AddV24uq__net_std_tf2_1/dense_7/kernel/Regularizer/add:z:06uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: W
IdentityIdentitySqrt:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall@^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpC^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2�
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp2�
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpBuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:J F
'
_output_shapes
:���������2

_user_specified_namex
�
�
(__inference_dense_7_layer_call_fn_918614

inputs
unknown:2

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_918280o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
�
C__inference_dense_6_layer_call_and_return_conditional_losses_918248

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�!
�
"__inference__traced_restore_918715
file_prefix'
assignvariableop_variable:E
3assignvariableop_1_uq__net_std_tf2_1_dense_6_kernel:22?
1assignvariableop_2_uq__net_std_tf2_1_dense_6_bias:2E
3assignvariableop_3_uq__net_std_tf2_1_dense_8_kernel:
?
1assignvariableop_4_uq__net_std_tf2_1_dense_8_bias:E
3assignvariableop_5_uq__net_std_tf2_1_dense_7_kernel:2
?
1assignvariableop_6_uq__net_std_tf2_1_dense_7_bias:


identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&custom_bias/.ATTRIBUTES/VARIABLE_VALUEB,inputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB*inputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB-outputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+outputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp3assignvariableop_1_uq__net_std_tf2_1_dense_6_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp1assignvariableop_2_uq__net_std_tf2_1_dense_6_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp3assignvariableop_3_uq__net_std_tf2_1_dense_8_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp1assignvariableop_4_uq__net_std_tf2_1_dense_8_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp3assignvariableop_5_uq__net_std_tf2_1_dense_7_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp1assignvariableop_6_uq__net_std_tf2_1_dense_7_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�*
�
!__inference__wrapped_model_918231
input_1J
8uq__net_std_tf2_1_dense_6_matmul_readvariableop_resource:22G
9uq__net_std_tf2_1_dense_6_biasadd_readvariableop_resource:2J
8uq__net_std_tf2_1_dense_7_matmul_readvariableop_resource:2
G
9uq__net_std_tf2_1_dense_7_biasadd_readvariableop_resource:
J
8uq__net_std_tf2_1_dense_8_matmul_readvariableop_resource:
G
9uq__net_std_tf2_1_dense_8_biasadd_readvariableop_resource:?
1uq__net_std_tf2_1_biasadd_readvariableop_resource:
identity��(uq__net_std_tf2_1/BiasAdd/ReadVariableOp�0uq__net_std_tf2_1/dense_6/BiasAdd/ReadVariableOp�/uq__net_std_tf2_1/dense_6/MatMul/ReadVariableOp�0uq__net_std_tf2_1/dense_7/BiasAdd/ReadVariableOp�/uq__net_std_tf2_1/dense_7/MatMul/ReadVariableOp�0uq__net_std_tf2_1/dense_8/BiasAdd/ReadVariableOp�/uq__net_std_tf2_1/dense_8/MatMul/ReadVariableOp�
/uq__net_std_tf2_1/dense_6/MatMul/ReadVariableOpReadVariableOp8uq__net_std_tf2_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
 uq__net_std_tf2_1/dense_6/MatMulMatMulinput_17uq__net_std_tf2_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
0uq__net_std_tf2_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp9uq__net_std_tf2_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
!uq__net_std_tf2_1/dense_6/BiasAddBiasAdd*uq__net_std_tf2_1/dense_6/MatMul:product:08uq__net_std_tf2_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
/uq__net_std_tf2_1/dense_7/MatMul/ReadVariableOpReadVariableOp8uq__net_std_tf2_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
 uq__net_std_tf2_1/dense_7/MatMulMatMul*uq__net_std_tf2_1/dense_6/BiasAdd:output:07uq__net_std_tf2_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
0uq__net_std_tf2_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp9uq__net_std_tf2_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
!uq__net_std_tf2_1/dense_7/BiasAddBiasAdd*uq__net_std_tf2_1/dense_7/MatMul:product:08uq__net_std_tf2_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
uq__net_std_tf2_1/dense_7/ReluRelu*uq__net_std_tf2_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
/uq__net_std_tf2_1/dense_8/MatMul/ReadVariableOpReadVariableOp8uq__net_std_tf2_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
 uq__net_std_tf2_1/dense_8/MatMulMatMul,uq__net_std_tf2_1/dense_7/Relu:activations:07uq__net_std_tf2_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0uq__net_std_tf2_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp9uq__net_std_tf2_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!uq__net_std_tf2_1/dense_8/BiasAddBiasAdd*uq__net_std_tf2_1/dense_8/MatMul:product:08uq__net_std_tf2_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(uq__net_std_tf2_1/BiasAdd/ReadVariableOpReadVariableOp1uq__net_std_tf2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
uq__net_std_tf2_1/BiasAddBiasAdd*uq__net_std_tf2_1/dense_8/BiasAdd:output:00uq__net_std_tf2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
uq__net_std_tf2_1/SquareSquare"uq__net_std_tf2_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������\
uq__net_std_tf2_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
uq__net_std_tf2_1/addAddV2uq__net_std_tf2_1/Square:y:0 uq__net_std_tf2_1/add/y:output:0*
T0*'
_output_shapes
:���������k
uq__net_std_tf2_1/SqrtSqrtuq__net_std_tf2_1/add:z:0*
T0*'
_output_shapes
:���������i
IdentityIdentityuq__net_std_tf2_1/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^uq__net_std_tf2_1/BiasAdd/ReadVariableOp1^uq__net_std_tf2_1/dense_6/BiasAdd/ReadVariableOp0^uq__net_std_tf2_1/dense_6/MatMul/ReadVariableOp1^uq__net_std_tf2_1/dense_7/BiasAdd/ReadVariableOp0^uq__net_std_tf2_1/dense_7/MatMul/ReadVariableOp1^uq__net_std_tf2_1/dense_8/BiasAdd/ReadVariableOp0^uq__net_std_tf2_1/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 2T
(uq__net_std_tf2_1/BiasAdd/ReadVariableOp(uq__net_std_tf2_1/BiasAdd/ReadVariableOp2d
0uq__net_std_tf2_1/dense_6/BiasAdd/ReadVariableOp0uq__net_std_tf2_1/dense_6/BiasAdd/ReadVariableOp2b
/uq__net_std_tf2_1/dense_6/MatMul/ReadVariableOp/uq__net_std_tf2_1/dense_6/MatMul/ReadVariableOp2d
0uq__net_std_tf2_1/dense_7/BiasAdd/ReadVariableOp0uq__net_std_tf2_1/dense_7/BiasAdd/ReadVariableOp2b
/uq__net_std_tf2_1/dense_7/MatMul/ReadVariableOp/uq__net_std_tf2_1/dense_7/MatMul/ReadVariableOp2d
0uq__net_std_tf2_1/dense_8/BiasAdd/ReadVariableOp0uq__net_std_tf2_1/dense_8/BiasAdd/ReadVariableOp2b
/uq__net_std_tf2_1/dense_8/MatMul/ReadVariableOp/uq__net_std_tf2_1/dense_8/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������2
!
_user_specified_name	input_1
�	
�
C__inference_dense_8_layer_call_and_return_conditional_losses_918296

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_dense_8_layer_call_fn_918560

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_918296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�"
�
C__inference_dense_7_layer_call_and_return_conditional_losses_918640

inputs0
matmul_readvariableop_resource:2
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp�Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/AbsAbsGuq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/SumSum4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/mulMul;uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/x:output:09uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/addAddV2;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const:output:04uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
3uq__net_std_tf2_1/dense_7/kernel/Regularizer/SquareSquareJuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1Sum7uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1Mul=uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/x:output:0;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1AddV24uq__net_std_tf2_1/dense_7/kernel/Regularizer/add:z:06uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp@^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpC^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp2�
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpBuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
�
2__inference_uq__net_std_tf2_1_layer_call_fn_918466
x
unknown:22
	unknown_0:2
	unknown_1:2

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *V
fQRO
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������2

_user_specified_namex
�	
�
C__inference_dense_8_layer_call_and_return_conditional_losses_918570

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_918590Z
Huq__net_std_tf2_1_dense_7_kernel_regularizer_abs_readvariableop_resource:2

identity��?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp�Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpw
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpHuq__net_std_tf2_1_dense_7_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:2
*
dtype0�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/AbsAbsGuq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/SumSum4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/mulMul;uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/x:output:09uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/addAddV2;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const:output:04uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHuq__net_std_tf2_1_dense_7_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:2
*
dtype0�
3uq__net_std_tf2_1/dense_7/kernel/Regularizer/SquareSquareJuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1Sum7uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1Mul=uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/x:output:0;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1AddV24uq__net_std_tf2_1/dense_7/kernel/Regularizer/add:z:06uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentity6uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp@^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpC^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp2�
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpBuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_dense_6_layer_call_fn_918541

inputs
unknown:22
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU 

XLA_CPU(2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_918248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�8
�
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918511
x8
&dense_6_matmul_readvariableop_resource:225
'dense_6_biasadd_readvariableop_resource:28
&dense_7_matmul_readvariableop_resource:2
5
'dense_7_biasadd_readvariableop_resource:
8
&dense_8_matmul_readvariableop_resource:
5
'dense_8_biasadd_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp�Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0t
dense_6/MatMulMatMulx%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
dense_7/MatMulMatMuldense_6/BiasAdd:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAdddense_8/BiasAdd:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T
SquareSquareBiasAdd:output:0*
T0*'
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=Z
addAddV2
Square:y:0add/y:output:0*
T0*'
_output_shapes
:���������G
SqrtSqrtadd:z:0*
T0*'
_output_shapes
:���������w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/AbsAbsGuq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/SumSum4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/mulMul;uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul/x:output:09uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
0uq__net_std_tf2_1/dense_7/kernel/Regularizer/addAddV2;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const:output:04uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype0�
3uq__net_std_tf2_1/dense_7/kernel/Regularizer/SquareSquareJuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
�
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1Sum7uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square:y:0=uq__net_std_tf2_1/dense_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1Mul=uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1/x:output:0;uq__net_std_tf2_1/dense_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
2uq__net_std_tf2_1/dense_7/kernel/Regularizer/add_1AddV24uq__net_std_tf2_1/dense_7/kernel/Regularizer/add:z:06uq__net_std_tf2_1/dense_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: W
IdentityIdentitySqrt:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp@^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOpC^uq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2�
?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp?uq__net_std_tf2_1/dense_7/kernel/Regularizer/Abs/ReadVariableOp2�
Buq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOpBuq__net_std_tf2_1/dense_7/kernel/Regularizer/Square/ReadVariableOp:J F
'
_output_shapes
:���������2

_user_specified_namex"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������2<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�E
�
configs
num_nodes_list

inputLayer
fcs
outputLayer
custom_bias
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
�
Max_iter
stop_losses

optimizers
lr
test_biases_list
num_neurons_mean
num_neurons_up
num_neurons_down
ypower_root_trans
plot_PI3NN_ylims"
trackable_dict_wrapper
 "
trackable_list_wrapper
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
'
!0"
trackable_list_wrapper
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
:2Variable
Q
0
1
*2
+3
"4
#5
6"
trackable_list_wrapper
Q
0
1
*2
+3
"4
#5
6"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_uq__net_std_tf2_1_layer_call_fn_918342
2__inference_uq__net_std_tf2_1_layer_call_fn_918466�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918511
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918432�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
!__inference__wrapped_model_918231input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
2serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
2:0222 uq__net_std_tf2_1/dense_6/kernel
,:*22uq__net_std_tf2_1/dense_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_6_layer_call_fn_918541�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_6_layer_call_and_return_conditional_losses_918551�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�

*kernel
+bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
2:0
2 uq__net_std_tf2_1/dense_8/kernel
,:*2uq__net_std_tf2_1/dense_8/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_8_layer_call_fn_918560�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_8_layer_call_and_return_conditional_losses_918570�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2:02
2 uq__net_std_tf2_1/dense_7/kernel
,:*
2uq__net_std_tf2_1/dense_7/bias
�2�
__inference_loss_fn_0_918590�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
 "
trackable_list_wrapper
5
0
!1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_918532input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_7_layer_call_fn_918614�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_7_layer_call_and_return_conditional_losses_918640�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
'
,0"
trackable_list_wrapper
 "
trackable_dict_wrapper�
!__inference__wrapped_model_918231p*+"#0�-
&�#
!�
input_1���������2
� "3�0
.
output_1"�
output_1����������
C__inference_dense_6_layer_call_and_return_conditional_losses_918551\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� {
(__inference_dense_6_layer_call_fn_918541O/�,
%�"
 �
inputs���������2
� "����������2�
C__inference_dense_7_layer_call_and_return_conditional_losses_918640\*+/�,
%�"
 �
inputs���������2
� "%�"
�
0���������

� {
(__inference_dense_7_layer_call_fn_918614O*+/�,
%�"
 �
inputs���������2
� "����������
�
C__inference_dense_8_layer_call_and_return_conditional_losses_918570\"#/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
(__inference_dense_8_layer_call_fn_918560O"#/�,
%�"
 �
inputs���������

� "����������;
__inference_loss_fn_0_918590*�

� 
� "� �
$__inference_signature_wrapper_918532{*+"#;�8
� 
1�.
,
input_1!�
input_1���������2"3�0
.
output_1"�
output_1����������
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918432b*+"#0�-
&�#
!�
input_1���������2
� "%�"
�
0���������
� �
M__inference_uq__net_std_tf2_1_layer_call_and_return_conditional_losses_918511\*+"#*�'
 �
�
x���������2
� "%�"
�
0���������
� �
2__inference_uq__net_std_tf2_1_layer_call_fn_918342U*+"#0�-
&�#
!�
input_1���������2
� "�����������
2__inference_uq__net_std_tf2_1_layer_call_fn_918466O*+"#*�'
 �
�
x���������2
� "����������