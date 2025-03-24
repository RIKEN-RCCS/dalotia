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
        use, intrinsic::ISO_C_BINDING, only: C_ptr, C_char
        implicit none
        character(kind=C_char), dimension(*), intent(in):: file_name
    end function dalotia_open_file_c

    subroutine dalotia_close_file(dalotia_file_pointer) bind(C,name="dalotia_close_file")
        use, intrinsic::ISO_C_BINDING, only: C_ptr
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
    end subroutine dalotia_close_file

    pure integer function dalotia_sizeof_weight_format(dalotia_weight_format) bind(C,name="dalotia_sizeof_weight_format")
        use, intrinsic::ISO_C_BINDING, only: C_int
        implicit none
        integer(C_int), intent(in), value:: dalotia_weight_format
    end function dalotia_sizeof_weight_format

    pure logical(C_bool) function dalotia_is_sparse_c(dalotia_file_pointer, tensor_name) bind(C,name="dalotia_is_sparse")
        use, intrinsic::ISO_C_BINDING, only: C_ptr, C_char, C_bool
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char), dimension(*), intent(in) :: tensor_name
    end function dalotia_is_sparse_c

    pure integer function dalotia_get_num_tensors(dalotia_file_pointer) bind(C,name="dalotia_get_num_tensors")
        use, intrinsic::ISO_C_BINDING, only: C_ptr
        implicit none
        type(C_ptr), intent(in), value :: dalotia_file_pointer
    end function dalotia_get_num_tensors

    integer function dalotia_get_tensor_name_c(dalotia_file_pointer, tensor_index_c, tensor_name) &
          bind(C,name="dalotia_get_tensor_name")
        use, intrinsic::ISO_C_binding, only: C_ptr, C_int, C_char
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        integer(C_int), intent(in), value:: tensor_index_c
        character(kind=C_char), dimension(*), intent(out):: tensor_name
    end function dalotia_get_tensor_name_c

    pure integer function dalotia_get_num_dimensions_c(dalotia_file_pointer, tensor_name) bind(C,name="dalotia_get_num_dimensions")
        use, intrinsic::ISO_C_binding, only: C_ptr, C_char
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char), dimension(*), intent(in) :: tensor_name
    end function dalotia_get_num_dimensions_c

    pure integer function dalotia_get_num_tensor_elements_c(dalotia_file_pointer, tensor_name) &
           bind(C,name="dalotia_get_num_tensor_elements")
        use, intrinsic::ISO_C_binding, only: C_ptr, C_char
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char), dimension(*), intent(in):: tensor_name
    end function dalotia_get_num_tensor_elements_c

    integer function dalotia_get_tensor_extents_c(dalotia_file_pointer, &
            tensor_name, tensor_extents) bind(C,name="dalotia_get_tensor_extents")
        use, intrinsic::ISO_C_binding, only: C_ptr, C_char, C_int
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char), dimension(*), intent(in):: tensor_name
        integer(C_int), dimension(*), intent(inout):: tensor_extents
    end function dalotia_get_tensor_extents_c

    subroutine dalotia_load_tensor_dense_c(dalotia_file_pointer, &
           tensor_name, tensor, dalotia_weight_format, dalotia_ordering) bind(C,name="dalotia_load_tensor_dense")
        use, intrinsic::ISO_C_binding, only: C_ptr, C_char, C_int
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char), dimension(*), intent(in):: tensor_name
        character(kind=C_char), dimension(*), intent(inout):: tensor
        integer(C_int), intent(in), value:: dalotia_weight_format
        integer(C_int), intent(in), value:: dalotia_ordering
    end subroutine dalotia_load_tensor_dense_c

    subroutine dalotia_load_tensor_dense_with_permutation_c( &
      dalotia_file_pointer, tensor_name, tensor, dalotia_weight_format, &
      dalotia_ordering, permutation) bind(C,name="dalotia_load_tensor_dense_with_permutation")
        use, intrinsic::ISO_C_BINDING, only: C_ptr, C_char, C_int
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char), dimension(*), intent(in):: tensor_name
        character(kind=C_char), dimension(*), intent(inout):: tensor
        integer(C_int), intent(in), value:: dalotia_weight_format
        integer(C_int), intent(in), value:: dalotia_ordering
        integer(C_int), dimension(*), intent(in):: permutation
    end subroutine dalotia_load_tensor_dense_with_permutation_c
  end interface

  interface dalotia_load_tensor_dense !TODO how many do we want in this interface? Codegen?
    module procedure dalotia_load_rank_1_float_tensor_dense
    module procedure dalotia_load_rank_1_double_tensor_dense
    module procedure dalotia_load_rank_2_float_tensor_dense
    module procedure dalotia_load_rank_2_double_tensor_dense
    module procedure dalotia_load_rank_3_float_tensor_dense
    module procedure dalotia_load_rank_3_double_tensor_dense
    module procedure dalotia_load_rank_4_float_tensor_dense
    module procedure dalotia_load_rank_4_double_tensor_dense
    module procedure dalotia_load_rank_5_float_tensor_dense
    module procedure dalotia_load_rank_5_double_tensor_dense
  end interface
  interface dalotia_load_tensor
    module procedure dalotia_load_rank_1_fixed_dim_tensor_dense
    module procedure dalotia_load_rank_2_fixed_dim_tensor_dense
    module procedure dalotia_load_rank_3_fixed_dim_tensor_dense
    module procedure dalotia_load_rank_4_fixed_dim_tensor_dense
    module procedure dalotia_load_rank_5_fixed_dim_tensor_dense
  end interface
  
  contains
    subroutine assert_expected_rank(tensor_rank, expected_rank)
        implicit none
        integer, intent(in):: tensor_rank, expected_rank
        if (tensor_rank /= expected_rank) then
            write (*, *) "dalotia: expected rank ", expected_rank, " but got ", tensor_rank
            error stop "unexpected rank" 
            ! disabling STOP, because apparently some compilers don't properly dispatch by rank
            ! in the interface block above, so the rank-1 is always called
        end if
    end subroutine assert_expected_rank

    subroutine assert_expected_extents(tensor_rank, tensor_extents, expected_extents)
        implicit none
        integer, intent(in):: tensor_rank
        integer, dimension(tensor_rank), intent(in):: tensor_extents
        integer, dimension(tensor_rank), intent(in):: expected_extents
        if (any(tensor_extents /= expected_extents)) then
            write (*, *) "dalotia: expected extents (", expected_extents, &
                         ") but got (", tensor_extents, ")"
            error stop "unexpected dalotia tensor extents" 
        end if
    end subroutine assert_expected_extents

    type(C_ptr) function dalotia_open_file(file_name)
        ! delegate to C function with trimmed name
        implicit none
        character(kind=C_char, len=*), intent(in):: file_name
        dalotia_open_file = dalotia_open_file_c(trim(file_name) // NUL)
    end function dalotia_open_file

    pure logical(C_bool) function dalotia_is_sparse(dalotia_file_pointer, tensor_name)
        ! delegate to C function with trimmed name
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char,len=*), intent(in) :: tensor_name
        dalotia_is_sparse = dalotia_is_sparse_c(dalotia_file_pointer, trim(tensor_name) // NUL)
    end function dalotia_is_sparse

    integer function dalotia_get_tensor_name(dalotia_file_pointer, tensor_index_fortran, tensor_name)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        integer, intent(in), value:: tensor_index_fortran
        integer(C_int):: tensor_index_c
        character(:,kind=C_char), allocatable, intent(out):: tensor_name
        character(:,kind=C_char), allocatable :: tensor_name_c
        integer :: tensor_name_length
        
        ! use C indexing
        tensor_index_c = tensor_index_fortran - 1
        allocate(character(kind=C_char,len=256) :: tensor_name_c)
        tensor_name_length = dalotia_get_tensor_name_c(dalotia_file_pointer, tensor_index_c, tensor_name_c)
        if (tensor_name_c(tensor_name_length + 1:tensor_name_length + 1) .ne. NUL) then
            !should not happen
            write (*, *) "not nul '", tensor_name_c(tensor_name_length + 1:tensor_name_length + 1)
        end if
        allocate(character(kind=C_char,len=tensor_name_length) :: tensor_name)
        tensor_name = tensor_name_c(1:tensor_name_length)

        ! set return value
        dalotia_get_tensor_name = tensor_name_length
    end function dalotia_get_tensor_name

    pure integer function dalotia_get_num_dimensions(dalotia_file_pointer, tensor_name)
        ! delegate to C function with trimmed name
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char,len=*), intent(in) :: tensor_name
        dalotia_get_num_dimensions = dalotia_get_num_dimensions_c(dalotia_file_pointer, trim(tensor_name) // NUL)
    end function dalotia_get_num_dimensions

    pure integer function dalotia_get_num_tensor_elements(dalotia_file_pointer, tensor_name)
        ! delegate to C function with trimmed name
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char,len=*), intent(in):: tensor_name
        dalotia_get_num_tensor_elements = dalotia_get_num_tensor_elements_c(dalotia_file_pointer, trim(tensor_name) // NUL)
    end function dalotia_get_num_tensor_elements

    subroutine dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, &
                                                tensor_rank, tensor_extents, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        integer(C_int), intent(in) :: tensor_rank !TODO deduce rank from tensor_extents
        integer(C_int), intent(out):: tensor_extents(tensor_rank)
        integer(C_int), optional, intent(in):: permutation(tensor_rank)
        character(kind=C_char, len=:), allocatable :: tensor_name_c
        integer(C_int) :: tensor_rank_returned

        allocate(character(kind=C_char, len=len(tensor_name) + 1) :: tensor_name_c)
        tensor_name_c = trim(tensor_name) // NUL !Appending null appears to be unnecessary in some cases
        
        tensor_rank_returned = dalotia_get_tensor_extents_c(dalotia_file_pointer, trim(tensor_name_c), tensor_extents)
        call assert_expected_rank(tensor_rank_returned, tensor_rank)
        ! reverse the order of the dimensions; Fortran is column-major
        tensor_extents = tensor_extents(tensor_rank:1:-1)

        if (present(permutation)) then
            tensor_extents(:) = tensor_extents(permutation(:))
        end if
    end subroutine dalotia_get_tensor_extents_fixed

    subroutine dalotia_get_tensor_extents(dalotia_file_pointer, tensor_name, tensor_extents, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        integer(C_int), allocatable, intent(out):: tensor_extents(:)
        integer(C_int) :: tensor_rank
        integer(C_int), dimension(:), optional, intent(in):: permutation
        character(kind=C_char, len=:), allocatable :: tensor_name_c

        allocate(character(kind=C_char, len=len(tensor_name) + 1) :: tensor_name_c)
        tensor_name_c = trim(tensor_name) // NUL !Appending null appears to be unnecessary in some cases
        
        tensor_rank = dalotia_get_num_dimensions(dalotia_file_pointer, tensor_name_c)
        allocate(tensor_extents(tensor_rank))
        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, &
                    trim(tensor_name_c), tensor_rank, tensor_extents, permutation)
    end subroutine dalotia_get_tensor_extents

    integer(C_int) function dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, &
      tensor_bytes, weight_format, permutation) result(num_tensor_elements)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        integer(C_int), intent(in) :: weight_format
        character(C_char), dimension(:), allocatable, target, intent(out):: tensor_bytes
        integer(C_int), dimension(:), optional, intent(in):: permutation
        integer(C_int) :: ordering, i

        num_tensor_elements = dalotia_get_num_tensor_elements(dalotia_file_pointer, tensor_name)

        allocate( tensor_bytes(num_tensor_elements * dalotia_sizeof_weight_format(weight_format)))
        if (present(permutation)) then
            ordering = dalotia_F_ordering
            call dalotia_load_tensor_dense_with_permutation_c(dalotia_file_pointer, trim(tensor_name) // NUL, &
                tensor_bytes, weight_format, ordering, permutation)
        else
            ordering = dalotia_C_ordering
            call dalotia_load_tensor_dense_c(dalotia_file_pointer, trim(tensor_name) // NUL, tensor_bytes, &
                 weight_format, ordering)
        end if
    end function dalotia_load_rank_1_byte_tensor_dense

    integer(kind=C_int) function get_dalotia_weight_format_from_kind(tensor_kind)
        use, intrinsic::ISO_Fortran_env, only: REAL32, REAL64
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

    subroutine dalotia_load_rank_1_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:), allocatable, intent(out):: tensor
        character(C_char), dimension(:), allocatable:: tensor_bytes
        integer(C_int), dimension(:), optional, intent(in):: permutation
        integer(C_int) :: num_tensor_elements

        num_tensor_elements = dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, &
                        get_dalotia_weight_format_from_kind(kind(tensor)), permutation)
        
        ! transfer into the real tensor
        ! cf. https://community.intel.com/t5/Intel-Fortran-Compiler/reinterpret-cast-for-arrays/td-p/855632
        tensor = transfer(tensor_bytes, tensor, num_tensor_elements)
    end subroutine dalotia_load_rank_1_float_tensor_dense

    subroutine dalotia_load_rank_1_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), dimension(:), allocatable, intent(out):: tensor
        character(C_char), dimension(:), allocatable:: tensor_bytes
        integer(C_int), dimension(:), optional, intent(in):: permutation
        integer(C_int) :: num_tensor_elements

        num_tensor_elements = dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, &
                get_dalotia_weight_format_from_kind(kind(tensor)), permutation)

        ! transfer into the real tensor
        ! cf. https://community.intel.com/t5/Intel-Fortran-Compiler/reinterpret-cast-for-arrays/td-p/855632
        tensor = transfer(tensor_bytes, tensor, num_tensor_elements)
    end subroutine dalotia_load_rank_1_double_tensor_dense

    subroutine dalotia_load_rank_1_fixed_dim_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        class(*), dimension(:), intent(out):: tensor
        character(C_char), dimension(:), allocatable:: tensor_bytes
        integer(C_int), dimension(:), optional, intent(in):: permutation
        integer(C_int) :: num_tensor_elements

        select type (tensor)
        type is (real(C_float))
            block
            num_tensor_elements = dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, &
                            get_dalotia_weight_format_from_kind(kind(tensor)), permutation)
            call assert_expected_extents(1, [num_tensor_elements], shape(tensor))
            ! transfer into the real tensor
            ! cf. https://community.intel.com/t5/Intel-Fortran-Compiler/reinterpret-cast-for-arrays/td-p/855632
            tensor = transfer(tensor_bytes, tensor, num_tensor_elements)
            end block
        type is (real(C_double))!TODO avoid exact duplication in blocks? https://stackoverflow.com/a/50550440
            block
            num_tensor_elements = dalotia_load_rank_1_byte_tensor_dense(dalotia_file_pointer, tensor_name, tensor_bytes, &
                            get_dalotia_weight_format_from_kind(kind(tensor)), permutation)
            call assert_expected_extents(1, [num_tensor_elements], shape(tensor))
            tensor = transfer(tensor_bytes, tensor, num_tensor_elements)
            end block
        class default
            error stop "dalotia fortran interface: unsupported tensor type"
        end select
    end subroutine dalotia_load_rank_1_fixed_dim_tensor_dense

    subroutine dalotia_load_rank_2_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        !TODO: is there a way to make this rank or type or assumed/allocatable agnostic / less code duplication?
        ! ranks: assumed-rank + select rank in fortran 2018? type: select type?
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        ! class(*), dimension(:,:), allocatable, intent(out) :: tensor
        real(C_float), dimension(:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(2)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(2)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 2, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        tensor = reshape(tensor_1d, tensor_extents)
        ! call C_F_POINTER (C_LOC(tensor_1d), tensor, tensor_extents)
    end subroutine dalotia_load_rank_2_float_tensor_dense

    subroutine dalotia_load_rank_2_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), dimension(:,:), allocatable, intent(out) :: tensor ! pointer,
        integer(C_int), optional, intent(in):: permutation(2)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d !, target
        integer(C_int) :: tensor_extents(2)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 2, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 2D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_2_double_tensor_dense

    subroutine dalotia_load_rank_2_fixed_dim_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        class(*), dimension(:, :), intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(2)
        integer(C_int) :: tensor_extents(2)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 2, tensor_extents, permutation)
        
        select type (tensor)
        type is (real(C_float))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(2, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        type is (real(C_double))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(2, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        class default
            error stop "dalotia fortran interface: unsupported tensor type"
        end select
    end subroutine dalotia_load_rank_2_fixed_dim_tensor_dense

    subroutine dalotia_load_rank_3_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:,:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(3)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(3)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 3, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 3D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_3_float_tensor_dense

    subroutine dalotia_load_rank_3_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), dimension(:,:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(3)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(3)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 3, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 3D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_3_double_tensor_dense

    subroutine dalotia_load_rank_3_fixed_dim_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        class(*), dimension(:, :, :), intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(3)
        integer(C_int) :: tensor_extents(3)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 3, tensor_extents, permutation)
        
        select type (tensor)
        type is (real(C_float))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(3, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        type is (real(C_double))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(3, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        class default
            error stop "dalotia fortran interface: unsupported tensor type"
        end select
    end subroutine dalotia_load_rank_3_fixed_dim_tensor_dense

    subroutine dalotia_load_rank_4_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:,:,:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(4)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(4)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 4, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 4D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_4_float_tensor_dense

    subroutine dalotia_load_rank_4_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), dimension(:,:,:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(4)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(4)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 4, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 4D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_4_double_tensor_dense

    subroutine dalotia_load_rank_4_fixed_dim_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        class(*), dimension(:, :, :, :), intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(4)
        integer(C_int) :: tensor_extents(4)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 4, tensor_extents, permutation)
        
        select type (tensor)
        type is (real(C_float))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(4, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        type is (real(C_double))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(4, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        class default
            error stop "dalotia fortran interface: unsupported tensor type"
        end select
    end subroutine dalotia_load_rank_4_fixed_dim_tensor_dense


    subroutine dalotia_load_rank_5_float_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_float), dimension(:,:,:,:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(5)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(5)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 5, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 5D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_5_float_tensor_dense

    subroutine dalotia_load_rank_5_double_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        real(C_double), dimension(:,:,:,:,:), allocatable, intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(5)
        real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
        integer(C_int) :: tensor_extents(5)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 5, tensor_extents, permutation)
        call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
        ! reshape into 5D tensor
        tensor = reshape(tensor_1d, tensor_extents)
    end subroutine dalotia_load_rank_5_double_tensor_dense

    subroutine dalotia_load_rank_5_fixed_dim_tensor_dense(dalotia_file_pointer, tensor_name, tensor, permutation)
        implicit none
        type(C_ptr), intent(in), value:: dalotia_file_pointer
        character(kind=C_char, len=*), intent(in):: tensor_name
        class(*), dimension(:, :, :, :, :), intent(out) :: tensor
        integer(C_int), optional, intent(in):: permutation(5)
        integer(C_int) :: tensor_extents(5)

        call dalotia_get_tensor_extents_fixed(dalotia_file_pointer, tensor_name, 5, tensor_extents, permutation)
        
        select type (tensor)
        type is (real(C_float))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(5, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        type is (real(C_double))
            block
            real(kind=kind(tensor)), dimension(:), allocatable:: tensor_1d
                call dalotia_load_tensor_dense(dalotia_file_pointer, tensor_name, tensor_1d, permutation)
                call assert_expected_extents(5, tensor_extents, shape(tensor))
                tensor = reshape(tensor_1d, shape(tensor))
            end block
        class default
            error stop "dalotia fortran interface: unsupported tensor type"
        end select
    end subroutine dalotia_load_rank_5_fixed_dim_tensor_dense

end module dalotia_c_interface