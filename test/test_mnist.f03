
program test_mnist
   use dalotia_c_interface   
  implicit none
    real :: images(28, 28, 10000)
    character(100) :: filename
    type(C_ptr) :: dalotia_file_pointer
    real(C_float), dimension(:,:,:,:), allocatable :: tensor_weight_conv1, tensor_weight_conv2, tensor_weight_4d_unused
    real(C_double), dimension(:,:), allocatable :: tensor_weight_fc1, tensor_weight_2d_unused
    real(C_float), dimension(:), allocatable :: tensor_bias_conv1, tensor_bias_conv2, tensor_bias_fc1

    filename = "../data/model-mnist.safetensors"

    call test_get_tensor_names(trim(filename))
    call test_load(filename, "conv1", tensor_weight_conv1, tensor_weight_2d_unused, tensor_bias_conv1)
    call test_load(filename, "conv2", tensor_weight_conv2, tensor_weight_2d_unused, tensor_bias_conv2)
    call test_load(filename, "fc1" , tensor_weight_4d_unused, tensor_weight_fc1, tensor_bias_fc1)

    ! test permutations
    dalotia_file_pointer = dalotia_open_file(filename)
    call dalotia_load_tensor_dense(dalotia_file_pointer, "conv1.weight", tensor_weight_4d_unused, permutation=[1, 2, 3, 4])
    call assert( all( tensor_weight_4d_unused .eq. tensor_weight_conv1))

    call dalotia_load_tensor_dense(dalotia_file_pointer, "conv1.weight", tensor_weight_4d_unused, permutation=[4, 2, 1, 3])
    call assert_equal_int(ubound(tensor_weight_4d_unused, 1), ubound(tensor_weight_conv1, 4))
    call assert_equal_int(ubound(tensor_weight_4d_unused, 2), ubound(tensor_weight_conv1, 2))
    call assert_equal_int(ubound(tensor_weight_4d_unused, 3), ubound(tensor_weight_conv1, 1))
    call assert_equal_int(ubound(tensor_weight_4d_unused, 4), ubound(tensor_weight_conv1, 3))
    tensor_weight_4d_unused = reshape(tensor_weight_4d_unused, shape=shape(tensor_weight_conv1), order=[4, 2, 1, 3])
    call assert( all( tensor_weight_4d_unused .eq. tensor_weight_conv1))

    call dalotia_load_tensor_dense(dalotia_file_pointer, "conv1.weight", tensor_weight_4d_unused, permutation=[3, 1, 4, 2])
    call assert_equal_int(ubound(tensor_weight_4d_unused, 1), ubound(tensor_weight_conv1, 3))
    call assert_equal_int(ubound(tensor_weight_4d_unused, 2), ubound(tensor_weight_conv1, 1))
    call assert_equal_int(ubound(tensor_weight_4d_unused, 3), ubound(tensor_weight_conv1, 4))
    call assert_equal_int(ubound(tensor_weight_4d_unused, 4), ubound(tensor_weight_conv1, 2))
    tensor_weight_4d_unused = reshape(tensor_weight_4d_unused, shape=shape(tensor_weight_conv1), order=[3, 1, 4, 2])
    call assert( all( tensor_weight_4d_unused .eq. tensor_weight_conv1))

    call dalotia_close_file(dalotia_file_pointer)
contains

!cf. https://stackoverflow.com/a/55376595
subroutine raise_exception(message)
  integer i
  character(len=*) message
  print *,message
  i=1
  i=1/(i-i)
end subroutine raise_exception

subroutine assert(condition)
  logical, intent(in) :: condition
  if (.not. condition) then
    call raise_exception("Assertion failed")
  end if
end subroutine assert

subroutine assert_equal(a, b)
  real, intent(in) :: a, b
  if (a /= b) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_equal

subroutine assert_equal_int(a, b)
  integer, intent(in) :: a, b
  if (a /= b) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_equal_int

subroutine assert_equal_strings(a, b)
  character(len=*), intent(in) :: a, b
  if (a /= b) then
    write (*, *) "Expected ", b, ", got ", a
    call assert(.false.)
  end if
end subroutine assert_equal_strings

subroutine assert_close(a, b)
  real, intent(in) :: a, b
  if (abs(a - b) > 1e-4) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_close

subroutine assert_close_df(a, b)
  real(kind=C_double), intent(in) :: a
  real, intent(in)::b
  if (abs(a - b) > 1e-4) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_close_df

subroutine test_get_tensor_names(filename)
    character(*), intent(in) :: filename
    character(len=:), allocatable :: tensor_name
    integer :: num_tensors, i, tensor_name_length
    type (C_ptr) :: dalotia_file_pointer
    logical :: is_sparse
    integer(C_int) :: num_dimensions

    dalotia_file_pointer = dalotia_open_file(trim(filename))

    num_tensors = dalotia_get_num_tensors(dalotia_file_pointer)
    if (num_tensors .ne. 6) then
        call raise_exception("Expected 6 tensors in model-mnist.safetensors")
    end if

    do i = 1, num_tensors
        tensor_name_length = dalotia_get_tensor_name(dalotia_file_pointer, i, tensor_name)
        if (i == 1) then
            call assert_equal_strings(trim(tensor_name),"conv1.bias")
        else if (i == 2) then
            call assert_equal_strings(trim(tensor_name),"conv1.weight")
        else if (i == 3) then
            call assert_equal_strings(trim(tensor_name),"conv2.bias")
        else if (i == 4) then
            call assert_equal_strings(trim(tensor_name),"conv2.weight")
        else if (i == 5) then
            call assert_equal_strings(trim(tensor_name),"fc1.bias")
        else if (i == 6) then
            call assert_equal_strings(trim(tensor_name),"fc1.weight")
        end if
        is_sparse = dalotia_is_sparse(dalotia_file_pointer, trim(tensor_name))
        call assert(.not. is_sparse)
        num_dimensions = dalotia_get_num_dimensions(dalotia_file_pointer, trim(tensor_name))
        call assert(num_dimensions > 0)
    end do

    call dalotia_close_file(dalotia_file_pointer)
end subroutine test_get_tensor_names

subroutine test_load(filename, tensor_name, tensor_weight_4d, tensor_weight_2d, tensor_bias)
    character(*), intent(in) :: filename, tensor_name
    character(:), allocatable :: tensor_name_weight, tensor_name_bias
    type(C_ptr) :: dalotia_file_pointer
    integer(C_int), dimension(:), allocatable :: tensor_extents_weight, tensor_extents_bias
    real(C_float), dimension(:,:,:,:), allocatable, intent(out) :: tensor_weight_4d
    real(C_double), dimension(:,:), allocatable, intent(out) :: tensor_weight_2d
    real(C_float), dimension(:), allocatable, intent(out) :: tensor_bias
    integer(C_int) :: num_dimensions_weight, num_dimensions_bias, num_elements_weight
    integer :: i

    tensor_name_weight = trim(tensor_name) // ".weight"
    tensor_name_bias = trim(tensor_name) // ".bias"

    dalotia_file_pointer = dalotia_open_file(trim(filename) // NUL)
    
    num_elements_weight = dalotia_get_num_tensor_elements(dalotia_file_pointer, tensor_name_weight);
    call dalotia_get_tensor_extents(dalotia_file_pointer, trim(tensor_name_weight), tensor_extents_weight)
    num_dimensions_weight = ubound(tensor_extents_weight, 1)
    call dalotia_get_tensor_extents(dalotia_file_pointer, trim(tensor_name_bias), tensor_extents_bias)
    num_dimensions_bias = ubound(tensor_extents_bias, 1)
    if (tensor_name == "conv1") then
        call assert(num_dimensions_weight == 4)
        call assert(num_elements_weight == 72)
        call assert(num_dimensions_bias == 1)
        call assert(tensor_extents_weight(4) == 8)
        call assert(tensor_extents_weight(3) == 1)
        call assert(tensor_extents_weight(2) == 3)
        call assert(tensor_extents_weight(1) == 3)
        call assert(tensor_extents_bias(1) == 8)
    else if (tensor_name == "conv2") then
        call assert(num_dimensions_weight == 4)
        call assert(num_dimensions_bias == 1)
        call assert(tensor_extents_weight(4) == 16)
        call assert(tensor_extents_weight(3) == 8)
        call assert(tensor_extents_weight(2) == 3)
        call assert(tensor_extents_weight(1) == 3)
    else if (tensor_name == "fc1") then
        call assert(num_dimensions_weight == 2)
        call assert(num_dimensions_bias == 1)
        call assert(tensor_extents_weight(2) == 10)
        call assert(tensor_extents_weight(1) == 784)
    else
        call raise_exception("Unknown tensor name")
    end if

    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_bias), tensor_bias) 

    ! check if the first, second, and last values are as expected
    if (tensor_name == "conv1") then
        call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_weight), tensor_weight_4d) 
        do i = 1, 4
            call assert_equal_int(ubound(tensor_weight_4d, i), tensor_extents_weight(i))
        end do
        call assert_close(tensor_weight_4d(1,1,1,1), 0.944823)
        call assert_close(tensor_weight_4d(2,1,1,1), 1.25045)
        ! call assert_close(tensor_weight_4d(8,1,3,3), 0.211111) !beware: no bounds checking
        call assert_close(tensor_weight_4d(3,3,1,8), 0.211111)
        call assert_close(tensor_bias(1), 0.1796)
        call assert_close(tensor_bias(8), 0.6550)
    else if (tensor_name == "conv2") then
        call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_weight), tensor_weight_4d) 
        do i = 1, 4
            call assert_equal_int(ubound(tensor_weight_4d, i), tensor_extents_weight(i))
        end do
        call assert_close(tensor_weight_4d(1,1,1,1), -0.79839)
        call assert_close(tensor_weight_4d(2,1,1,1), -1.3640)
        call assert_close(tensor_weight_4d(3,3,8,16), 0.32985)
        call assert_close(tensor_bias(1), -0.2460)
        call assert_close(tensor_bias(16), -0.3158)
    else if (tensor_name == "fc1") then
        call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_weight), tensor_weight_2d) 
        do i = 1, 2
            call assert_equal_int(ubound(tensor_weight_2d, i), tensor_extents_weight(i))
        end do
        call assert_close_df(tensor_weight_2d(1,1), 0.3420)
        call assert_close_df(tensor_weight_2d(2,1), 0.7881)
        call assert_close_df(tensor_weight_2d(784,10), 0.3264)
        call assert_close(tensor_bias(1), 0.3484)
        call assert_close(tensor_bias(10), -0.2224)
    else 
        call raise_exception("Unknown tensor name")
    end if
    
    call dalotia_close_file(dalotia_file_pointer)
    write(*,*) "All loads are correct "
end subroutine test_load

end program test_mnist