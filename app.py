import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import svd, eigvals, eig

st.title('SIMPLE VECTOR MATRIX APPS')

with st.sidebar:
    tipe = st.radio('Pilih Tipe', ['single vector', 'double vector', 'single matrix', 'double matrix', 'Eigen', 'OBE', 'SVD', 'Quadratic Matrix'])

with st.expander('Pilih Ukuran'):
    with st.form('Pilih Ukuran'):
        if tipe == 'single vector':
            size = st.number_input('ukuran vektor', min_value=2)
        elif tipe == 'double matrix':
            row1 = st.number_input('ukuran baris matrix pertama', min_value=2)
            col1 = st.number_input('ukuran kolom matrix pertama', min_value=2)
            row2 = st.number_input('ukuran baris matrix kedua', min_value=2)
            col2 = st.number_input('ukuran kolom matrix kedua', min_value=2)
        elif tipe == 'double vector':
            size = st.number_input('ukuran double vector', min_value=2)
        submit = st.form_submit_button('submit size')

if tipe == 'single vector':
    df = pd.DataFrame(columns=range(1, size + 1), index=range(1, 2), dtype=float)
    st.write('Masukkan data untuk vektor')
    df_input = st.experimental_data_editor(df, use_container_width=True)

elif tipe == 'double vector':
    df = pd.DataFrame(columns=range(1, size + 1), index=range(1, 3), dtype=float)
    st.write('Masukkan data untuk double vector')
    df_input = st.experimental_data_editor(df, use_container_width=True)

elif tipe == 'single matrix':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_data_editor(df, use_container_width=True)
    st.write('Matrix:')
    st.write(df)

    if submit:  # Tambahkan kondisi ini
        # Operasi atau manipulasi pada matrix
        # Tambahkan kode sesuai dengan kebutuhan Anda
        Operasi = st.radio('Pilih Operasi', ['A*B', 'A+B', 'Determinan', 'Invers', 'Turunan'])
        matrix1 = df.fillna(0).to_numpy()
        if Operasi == 'A*B':
            # Lakukan operasi perkalian matriks dengan dirinya sendiri
            result = np.matmul(matrix1, matrix1)
            st.write(result)

        elif Operasi == 'A+B':
            # Lakukan operasi penjumlahan matriks dengan dirinya sendiri
            result = matrix1 + matrix1
            st.write(result)

        elif Operasi == 'Determinan':
            # Cari determinan matriks
            determinant = np.linalg.det(matrix1)
            st.write('Determinan:')
            st.write(determinant)

        elif Operasi == 'Turunan':
            try:
                derivative = np.gradient(matrix1)
                st.write('Matrix Derivative:')
                st.write(derivative)
            except ValueError:
                st.write('Operasi turunan tidak dapat dilakukan pada matrix ini.')


elif tipe == 'double matrix':
    df1 = pd.DataFrame(columns=range(1, col1 + 1), index=range(1, row1 + 1), dtype=float)
    df2 = pd.DataFrame(columns=range(1, col2 + 1), index=range(1, row2 + 1), dtype=float)
    st.write('Masukkan data untuk matrix pertama')
    df1_input = st.experimental_data_editor(df1, use_container_width=True)
    st.write('Masukkan data untuk matrix kedua')
    df2_input = st.experimental_data_editor(df2, use_container_width=True)
    st.write('Matrix Pertama:')
    st.write(df1)
    st.write('Matrix Kedua:')
    st.write(df2)

elif tipe == 'Eigen':
    st.write('1. Eigenvalue dan Eigenvector')
    st.write('2. Eigenvalues')
    st.write('3. Eigenvectors')
    option = st.number_input('Pilih pilihan yang diinginkan', min_value=1, max_value=3, value=1)
    if option == 1:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        st.write('Eigenvalues:')
        st.write(eigenvalues)
        st.write('Eigenvectors:')
        st.write(eigenvectors)

    elif option == 2:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        eigenvalues = np.linalg.eigvals(matrix)
        st.write('Eigenvalues:')
        st.write(eigenvalues)

    elif option == 3:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        eigenvectors = np.linalg.eig(matrix)[1]
        st.write('Eigenvectors:')
        st.write(eigenvectors)

elif tipe == 'OBE':
    st.write('1. Reduced Row Echelon Form')
    st.write('2. Row Echelon Form')
    option = st.number_input('Pilih pilihan yang diinginkan', min_value=1, max_value=2, value=1)
    if option == 1:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        rref = np.linalg.matrix_rank(matrix)
        st.write('Reduced Row Echelon Form:')
        st.write(rref)

    elif option == 2:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        ref = np.linalg.matrix_rank(matrix, hermitian=False)
        st.write('Row Echelon Form:')
        st.write(ref)

elif tipe == 'SVD':
    st.write('1. Singular Value Decomposition (SVD)')
    st.write('2. Eigenvalue Decomposition (EVD)')
    option = st.number_input('Pilih pilihan yang diinginkan', min_value=1, max_value=2, value=1)
    if option == 1:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        u, s, vh = np.linalg.svd(matrix)
        st.write('U:')
        st.write(u)
        st.write('Singular Values:')
        st.write(s)
        st.write('V:')
        st.write(vh.T)

    elif option == 2:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        st.write('Eigenvalues:')
        st.write(eigenvalues)
        st.write('Eigenvectors:')
        st.write(eigenvectors)

elif tipe == 'Quadratic Matrix':
    st.write('1. Quadratic Form')
    st.write('2. Positive Definite Matrix')
    option = st.number_input('Pilih pilihan yang diinginkan', min_value=1, max_value=2, value=1)
    if option == 1:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        vector = st.text_area('Masukkan vektor (dalam format numpy array)')
        vector = np.array(eval(vector))
        quadratic_form = np.dot(np.dot(vector.T, matrix), vector)
        st.write('Quadratic Form:')
        st.write(quadratic_form)

    elif option == 2:
        matrix = st.text_area('Masukkan matriks (dalam format numpy array)')
        matrix = np.array(eval(matrix))
        positive_definite = np.all(eigvals(matrix) > 0)
        st.write('Positive Definite Matrix:')
        st.write(positive_definite)
