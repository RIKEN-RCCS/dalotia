module dalotia_c_interface !TODO move to src
  ! cf. https://fortranwiki.org/fortran/show/c_interface_module
  use, intrinsic :: ISO_C_Binding, &
  ! C type aliases for pointer derived types:
      C_ptr => C_ptr , &
      C_char_ptr => C_ptr, &
      C_const_char_ptr => C_ptr, &
      C_void_ptr => C_ptr, &
      C_const_void_ptr => C_ptr
    implicit none
    character(len=1,kind=C_char), parameter :: NUL = C_NULL_char
  interface
    type(C_ptr) function dalotia_open_file(file_name) bind(C,name="open_file")
        use,intrinsic::ISO_C_BINDING
        implicit none
        character(kind=c_char),intent(in), value:: file_name
    end function dalotia_open_file

    subroutine dalotia_close_file(dalotia_file_pointer) bind(C,name="close_file")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value:: dalotia_file_pointer
    end subroutine dalotia_close_file

    logical function dalotia_is_sparse(dalotia_file_pointer, tensor_name) bind(C,name="is_sparse")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value:: dalotia_file_pointer
        character(kind=c_char),intent(in), value:: tensor_name
    end function dalotia_is_sparse

    integer function dalotia_get_num_tensors(dalotia_file_pointer) bind(C,name="get_num_tensors")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value :: dalotia_file_pointer
    end function dalotia_get_num_tensors

    integer function dalotia_get_tensor_name_c(dalotia_file_pointer, tensor_index_c, tensor_name) bind(C,name="get_tensor_name")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value:: dalotia_file_pointer
        integer(C_INT),intent(in), value:: tensor_index_c
        character(kind=c_char), dimension(*), intent(out):: tensor_name
    end function dalotia_get_tensor_name_c
  end interface
  contains
    integer function dalotia_get_tensor_name(dalotia_file_pointer, tensor_index_fortran, tensor_name)
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value:: dalotia_file_pointer
        integer,intent(in), value:: tensor_index_fortran
        integer(C_INT):: tensor_index_c
        character(kind=c_char), dimension(255), intent(out):: tensor_name
        integer :: i, tensor_name_length
        
        ! use C indexing
        tensor_index_c = tensor_index_fortran - 1
        tensor_name_length = dalotia_get_tensor_name_c(dalotia_file_pointer, tensor_index_c, tensor_name)
        if (tensor_name(tensor_name_length + 1) .ne. NUL) then
            !should not happen
            write (*, *) "not nul '", tensor_name(tensor_name_length + 1)
        end if
        ! fill with blank spaces
        i = tensor_name_length + 1
        do while(i<=size(tensor_name))
         tensor_name(i) = ' '
         i = i + 1
        end do
        dalotia_get_tensor_name = tensor_name_length
    end function dalotia_get_tensor_name
end module dalotia_c_interface

program test_mnist
   use dalotia_c_interface   
  implicit none
    real :: images(28, 28, 10000)
    character(100) :: filename

    filename = "../data/model-mnist.safetensors"

    call test_get_tensor_names(trim(filename))
    ! test_load(filename, "conv1"); !TODO
    ! test_load(filename, "conv2");
    ! test_load(filename, "fc1");

    stop ! early return for CI, to avoid data handling ;)

    images = read_mnist_scaled(trim("../../python-fiddling/dataset/MNIST/raw/t10k-images-idx3-ubyte"))

    ! write (*, *) images(:, :, 1)
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

subroutine assert_equal_strings(a, b)
  character(len=*), intent(in) :: a, b
  if (a /= b) then
    write (*, *) "Expected ", b, ", got ", a
    call assert(.false.)
  end if
end subroutine assert_equal_strings

subroutine test_get_tensor_names(filename)
    character(*), intent(in) :: filename
    character(255) :: tensor_name
    integer :: num_tensors, i, tensor_name_length
    type (C_ptr) :: dalotia_file_pointer
    logical :: is_sparse

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
    end do

    call dalotia_close_file(dalotia_file_pointer)
end subroutine test_get_tensor_names

function read_mnist_scaled(full_path) result (array_of_images)
    use, intrinsic :: iso_fortran_env, only: INT32, INT8
    implicit none

    character(*), intent(in) :: full_path
    real, dimension(28, 28, 10000) :: array_of_images
    
    integer(INT32) :: magic_number
    integer(INT32) :: number_of_images
    integer(INT32) :: n_rows
    integer(INT32) :: n_columns
    integer(INT8)  :: images_int8(28, 28, 10000)
    integer :: file_handle
    
    open (newunit = file_handle, file = full_path, action = 'read', form = 'unformatted', &
    &     access = 'stream', status = 'old', convert = 'big_endian')
    
    read (file_handle) magic_number, number_of_images, n_rows, n_columns, images_int8
    close (file_handle)
    
    array_of_images = real(iand(int(images_int8), 255)) / 255.0  ! unsigned 8-bit integer -> default integer -> real
end function read_mnist_scaled


end program test_mnist