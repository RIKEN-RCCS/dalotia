module dalotia_c_interface
  ! cf. https://fortranwiki.org/fortran/show/c_interface_module
  use, intrinsic :: ISO_C_binding, &
  ! C type aliases for pointer derived types:
      C_ptr => C_ptr , &
      C_char_ptr => C_ptr, &
      C_const_char_ptr => C_ptr, &
      C_void_ptr => C_ptr, &
      C_const_void_ptr => C_ptr
    implicit none
    character(len=1,kind=C_char), parameter :: NUL = C_NULL_char

  ! TODO which is the best C-enum syntax?
    enum, bind(C)
        enumerator dalotia_float_64  , &
                   dalotia_float_32  , &
                   dalotia_float_16  , &
                   dalotia_float_8   , &
                   dalotia_bfloat_16 , &
                   dalotia_int_8     , &
                   dalotia_int_2 
    end enum 

    enum, bind(C)
        enumerator dalotia_C_ordering, &
                   dalotia_F_ordering
    end enum

  interface
    type(C_ptr) function dalotia_open_file_c(file_name) bind(C,name="dalotia_open_file")
        use, intrinsic::ISO_C_BINDING
        implicit none
        character(kind=c_char), dimension(*), intent(in):: file_name
    end function dalotia_open_file_c

    subroutine dalotia_close_file(dalotia_file_pointer) bind(C,name="dalotia_close_file")
        use, intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
    end subroutine dalotia_close_file

    pure integer function dalotia_sizeof_weight_format(dalotia_weight_format) bind(C,name="dalotia_sizeof_weight_format")
        use, intrinsic::ISO_C_BINDING
        implicit none
        integer(C_int), intent(in), value:: dalotia_weight_format
    end function dalotia_sizeof_weight_format

    pure logical function dalotia_is_sparse_c(dalotia_file_pointer, tensor_name) bind(C,name="dalotia_is_sparse")
        use, intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char), dimension(*), intent(in) :: tensor_name
    end function dalotia_is_sparse_c

    pure integer function dalotia_get_num_tensors(dalotia_file_pointer) bind(C,name="dalotia_get_num_tensors")
        use, intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr), intent(in), value :: dalotia_file_pointer
    end function dalotia_get_num_tensors

    integer function dalotia_get_tensor_name_c(dalotia_file_pointer, tensor_index_c, tensor_name) &
          bind(C,name="dalotia_get_tensor_name")
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        integer(C_int), intent(in), value:: tensor_index_c
        character(kind=c_char), dimension(*), intent(out):: tensor_name
    end function dalotia_get_tensor_name_c

    pure integer function dalotia_get_num_dimensions_c(dalotia_file_pointer, tensor_name) bind(C,name="dalotia_get_num_dimensions")
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char), dimension(*), intent(in) :: tensor_name
    end function dalotia_get_num_dimensions_c

    pure integer function dalotia_get_num_tensor_elements_c(dalotia_file_pointer, tensor_name) &
           bind(C,name="dalotia_get_num_tensor_elements")
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char), dimension(*), intent(in):: tensor_name
    end function dalotia_get_num_tensor_elements_c

    integer function dalotia_get_tensor_extents_c(dalotia_file_pointer, &
            tensor_name, tensor_extents) bind(C,name="dalotia_get_tensor_extents")
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char), dimension(*), intent(in):: tensor_name
        integer(C_int), dimension(*), intent(inout):: tensor_extents
    end function dalotia_get_tensor_extents_c

    subroutine dalotia_load_tensor_dense_c(dalotia_file_pointer, &
           tensor_name, tensor, dalotia_weight_format, dalotia_ordering) bind(C,name="dalotia_load_tensor_dense")
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char), dimension(*), intent(in):: tensor_name
        character(kind=C_char), dimension(*), intent(inout):: tensor
        integer(C_int), intent(in), value:: dalotia_weight_format
        integer(C_int), intent(in), value:: dalotia_ordering
    end subroutine dalotia_load_tensor_dense_c
  end interface

  interface dalotia_load_tensor_dense
    module procedure dalotia_load_rank_1_float_tensor_dense
    module procedure dalotia_load_rank_1_double_tensor_dense
    module procedure dalotia_load_rank_2_float_tensor_dense
    module procedure dalotia_load_rank_2_double_tensor_dense
    module procedure dalotia_load_rank_3_float_tensor_dense
    module procedure dalotia_load_rank_4_float_tensor_dense
  end interface
  
  contains
    type(C_ptr) function dalotia_open_file(file_name)
        ! delegate to C function with trimmed name
        use, intrinsic::ISO_C_BINDING
        implicit none
        character(kind=c_char, len=*), intent(in):: file_name
        dalotia_open_file = dalotia_open_file_c(trim(file_name))
    end function dalotia_open_file

    pure logical function dalotia_is_sparse(dalotia_file_pointer, tensor_name)
        ! delegate to C function with trimmed name
        use, intrinsic::ISO_C_BINDING
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char,len=*), intent(in) :: tensor_name
        dalotia_is_sparse = dalotia_is_sparse_c(dalotia_file_pointer, trim(tensor_name))
    end function dalotia_is_sparse

    integer function dalotia_get_tensor_name(dalotia_file_pointer, tensor_index_fortran, tensor_name)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        integer, intent(in), value:: tensor_index_fortran
        integer(C_INT):: tensor_index_c
        character(:,kind=c_char), allocatable, intent(out):: tensor_name
        character(:,kind=c_char), allocatable :: tensor_name_c
        integer :: tensor_name_length
        
        ! use C indexing
        tensor_index_c = tensor_index_fortran - 1
        allocate(character(kind=c_char,len=256) :: tensor_name_c)
        tensor_name_length = dalotia_get_tensor_name_c(dalotia_file_pointer, tensor_index_c, tensor_name_c)
        if (tensor_name_c(tensor_name_length + 1:tensor_name_length + 1) .ne. NUL) then
            !should not happen
            write (*, *) "not nul '", tensor_name_c(tensor_name_length + 1:tensor_name_length + 1)
        end if
        allocate(character(kind=c_char,len=tensor_name_length) :: tensor_name)
        tensor_name = tensor_name_c(1:tensor_name_length)

        ! set return value
        dalotia_get_tensor_name = tensor_name_length
    end function dalotia_get_tensor_name

 pure integer function dalotia_get_num_dimensions(dalotia_file_pointer, tensor_name)
        ! delegate to C function with trimmed name
        use, intrinsic::ISO_C_binding, only: C_ptr, C_char
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char,len=*), intent(in) :: tensor_name
        dalotia_get_num_dimensions = dalotia_get_num_dimensions_c(dalotia_file_pointer, trim(tensor_name))
    end function dalotia_get_num_dimensions

    pure integer function dalotia_get_num_tensor_elements(dalotia_file_pointer, tensor_name)
        ! delegate to C function with trimmed name
        use, intrinsic::ISO_C_binding, only: C_ptr, C_char
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=c_char,len=*), intent(in):: tensor_name
        dalotia_get_num_tensor_elements = dalotia_get_num_tensor_elements_c(dalotia_file_pointer, trim(tensor_name))
    end function dalotia_get_num_tensor_elements

    subroutine dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        integer(C_int), allocatable, intent(out):: tensor_extents(:)
        integer(C_int) :: tensor_rank
        ! character(kind=c_char, len=:), allocatable :: tensor_name_c
        ! tensor_name_c = trim(tensor_name) // NUL !Appending null appears to be unnecessary
        
        tensor_rank = dalotia_get_num_dimensions(dalotia_file_pointer, tensor_name)
        allocate(tensor_extents(tensor_rank))
        tensor_rank = dalotia_get_tensor_extents_c(dalotia_file_pointer, trim(tensor_name), tensor_extents)
        ! assert(tensor_rank == ubound(tensor_extents))

        ! reverse the order of the dimensions; Fortran is column-major
        tensor_extents = tensor_extents(tensor_rank:1:-1)
    end subroutine dalotia_get_tensor_extents

    subroutine dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, weight_format)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        integer(C_int) :: num_tensor_elements, ordering
        integer(C_int), intent(in) :: weight_format
        character(C_char), dimension(:), allocatable, target, intent(out):: tensor_bytes

        num_tensor_elements = dalotia_get_num_tensor_elements(dalotia_file_pointer, tensor_name)

        ordering = dalotia_C_ordering
        allocate( tensor_bytes(num_tensor_elements * dalotia_sizeof_weight_format(weight_format)))
        call dalotia_load_tensor_dense_c(dalotia_file_pointer, trim(tensor_name), tensor_bytes, &
                 weight_format, ordering) !TODO add version that takes permutation and F_ordering
    end subroutine dalotia_load_rank_1_byte_tensor_dense

    integer(kind=C_int) function get_dalotia_weight_format_from_kind(tensor_kind)
        use, intrinsic::ISO_C_binding
        use, intrinsic::ISO_Fortran_env
        implicit none
        integer, intent(in) :: tensor_kind
        if (tensor_kind == C_float .or. tensor_kind == REAL32) then
            get_dalotia_weight_format_from_kind = dalotia_float_32
        else if (tensor_kind == C_double .or. tensor_kind == REAL64) then
            get_dalotia_weight_format_from_kind = dalotia_float_64
        else
            ! call raise_exception("dalotia fortran interface: unsupported tensor type")
            stop "dalotia fortran interface: unsupported tensor type"
        end if
    end function get_dalotia_weight_format_from_kind

    subroutine dalotia_load_rank_1_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), allocatable, intent(out):: tensor(:)
        character(C_char), dimension(:), allocatable:: tensor_bytes
        integer(C_int), allocatable:: tensor_extents(:)

        call dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        ! assert(tensor_rank == 1)
        call dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, &
                get_dalotia_weight_format_from_kind(kind(tensor)))

        ! transfer into the real tensor
        ! cf. https://community.intel.com/t5/Intel-Fortran-Compiler/reinterpret-cast-for-arrays/td-p/855632
        tensor = transfer(tensor_bytes, tensor, product(tensor_extents))
    end subroutine dalotia_load_rank_1_float_tensor_dense

    subroutine dalotia_load_rank_1_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), allocatable, intent(out):: tensor(:)
        character(C_char), dimension(:), allocatable:: tensor_bytes
        integer(C_int), allocatable:: tensor_extents(:)

        call dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        ! assert(tensor_rank == 1)
        call dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, &
                get_dalotia_weight_format_from_kind(kind(tensor)))

        ! transfer into the real tensor
        ! cf. https://community.intel.com/t5/Intel-Fortran-Compiler/reinterpret-cast-for-arrays/td-p/855632
        tensor = transfer(tensor_bytes, tensor, product(tensor_extents))
    end subroutine dalotia_load_rank_1_double_tensor_dense

    subroutine dalotia_load_rank_2_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor)
        !TODO: is there a way to make this rank agnostic / less code duplication?
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:,:), allocatable, intent(out) :: tensor
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int), dimension(:), allocatable:: tensor_extents
        integer(C_int), dimension(2) :: fixed_tensor_extents

        call dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        ! assert(tensor_rank == 2)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d)
        ! reshape into 2D tensor
        fixed_tensor_extents = [tensor_extents(1), tensor_extents(2)]
        tensor = reshape(tensor_1d, fixed_tensor_extents)
    end subroutine dalotia_load_rank_2_float_tensor_dense

    subroutine dalotia_load_rank_2_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), dimension(:,:), allocatable, intent(out) :: tensor
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int), dimension(:), allocatable:: tensor_extents
        integer(C_int), dimension(2) :: fixed_tensor_extents

        call dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        ! assert(tensor_rank == 2)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d)
        ! reshape into 2D tensor
        fixed_tensor_extents = [tensor_extents(1), tensor_extents(2)]
        tensor = reshape(tensor_1d, fixed_tensor_extents)
    end subroutine dalotia_load_rank_2_double_tensor_dense

    subroutine dalotia_load_rank_3_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:,:,:), allocatable, intent(out) :: tensor
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int), dimension(:), allocatable:: tensor_extents
        integer(C_int), dimension(3) :: fixed_tensor_extents

        call dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        ! assert(tensor_rank == 3)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d)
        ! reshape into 3D tensor
        fixed_tensor_extents = [tensor_extents(1), tensor_extents(2), tensor_extents(3)]
        tensor = reshape(tensor_1d, fixed_tensor_extents)
    end subroutine dalotia_load_rank_3_float_tensor_dense

    subroutine dalotia_load_rank_4_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor)
        use, intrinsic::ISO_C_binding
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:,:,:,:), allocatable, intent(out) :: tensor
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int), dimension(:), allocatable:: tensor_extents
        integer(C_int), dimension(4) :: fixed_tensor_extents

        call dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents)
        ! assert(tensor_rank == 4)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d)
        ! reshape into 4D tensor
        fixed_tensor_extents = [tensor_extents(1), tensor_extents(2), tensor_extents(3), tensor_extents(4)]
        tensor = reshape(tensor_1d, fixed_tensor_extents)
    end subroutine dalotia_load_rank_4_float_tensor_dense
end module dalotia_c_interface