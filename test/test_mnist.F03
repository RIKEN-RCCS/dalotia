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
  interface
    type(C_ptr) function dalotia_open_file(file_name) bind(C,name="open_file")
        use,intrinsic::ISO_C_BINDING
        implicit none
        character(kind=c_char),intent(in):: file_name
    end function dalotia_open_file

    subroutine dalotia_close_file(dalotia_file_pointer) bind(C,name="close_file")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value:: dalotia_file_pointer
        ! integer, pointer :: dalotia_file_pointer
    end subroutine dalotia_close_file

    integer function dalotia_get_num_tensors(dalotia_file_pointer) bind(C,name="get_num_tensors")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value :: dalotia_file_pointer
        ! integer, pointer :: dalotia_file_pointer
    end function dalotia_get_num_tensors

    subroutine dalotia_get_tensor_name(dalotia_file_pointer, tensor_index, tensor_name) bind(C,name="get_tensor_name")
        use,intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr),intent(in), value:: dalotia_file_pointer
        integer(C_INT),intent(in):: tensor_index
        character(kind=c_char),dimension(*),intent(out):: tensor_name
    end subroutine dalotia_get_tensor_name
  end interface
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

subroutine test_get_tensor_names(filename)
    character(*), intent(in) :: filename
    character(100) :: tensor_name
    integer :: num_tensors, i
    type (C_ptr) :: dalotia_file_pointer

    dalotia_file_pointer = dalotia_open_file(trim(filename))

    num_tensors = dalotia_get_num_tensors(dalotia_file_pointer)
    if (num_tensors .ne. 6) then
        call raise_exception("Expected 6 tensors in model-mnist.safetensors")
    end if

    ! do i = 1, num_tensors
    !     call dalotia_get_tensor_name(dalotia_file_pointer, i, tensor_name)
    !     if (i == 1) then
    !         if (trim(tensor_name) /= "konv1") then
    !             call raise_exception("Expected tensor name conv1, got " // trim(tensor_name))
    !         end if !TODO
    !     end if
    ! end do

    write (*, *) "OK??"
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