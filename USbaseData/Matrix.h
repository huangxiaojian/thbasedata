/////////////////////////////////////////////////////////////////////////////////
///  Copyright (C), 2015-?, Xiaojian Huang
///  \file  Matrix.h
///  \brief Matrix module in SIMD library
///  \details Matrix related functions and SSE and AVX instruction micro
///  \author Xiaojian Huang
///   \version 1.0
///   \date    2015.5
///   \History: 
///                
///      Xiaojian Huang   2015.5     1.0     Create this file 
///  \warning	All rights reserved.
///  Redistribution and use in source and binary forms, with or without
///  modification, are permitted provided that the following conditions
///  are met:
///     1. Redistributions of source code must retain the above copyright
///        notice, this list of conditions and the following disclaimer.
///     2. Redistributions in binary form must reproduce the above copyright
///        notice, this list of conditions and the following disclaimer in the
///        documentation and/or other materials provided with the distribution.
///     3. The names of its contributors may not be used to endorse or promote
///        products derived from this software without specific prior written
///        permission.
///  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
///  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
///  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
///  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
///  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
///  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
///  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
///  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
///  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
///	 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
///	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////
#ifndef TH_MATRIX_H
#define TH_MATRIX_H

#include "Vector.h"
#include "Quaternion.h"

ATTRIBUTE_ALIGNED(class) Matrix3x4
{
	///Data storage for the matrix, each vector is a row of the matrix
	Vector4 m_el[3];

public:
	/** @brief No initialization constructor */
	Matrix3x4(){}

	/** @brief Constructor with row major formatting */
	Matrix3x4(const Scalar& xx, const Scalar& xy, const Scalar& xz,
		const Scalar& yx, const Scalar& yy, const Scalar& yz,
		const Scalar& zx, const Scalar& zy, const Scalar& zz)
	{
		setValue(xx, xy, xz, yx, yy, yz, zx, zy, zz);
	}

	/** @brief Copy constructor */
	Matrix3x4(const Matrix3x4& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
	}

	/**@brief Constructor from Quaternion */
	explicit Matrix3x4(const Quaternion& q) { setRotation(q); }

	Matrix3x4(const Scalar4& e0, const Scalar4& e1, const Scalar4& e2)
	{
		m_el[0] = e0;
		m_el[1] = e1;
		m_el[2] = e2;
	}

	Matrix3x4(const Vector4& e0, const Vector4& e1, const Vector4& e2)
	{
		m_el[0] = e0;
		m_el[1] = e1;
		m_el[2] = e2;
	}

	friend Matrix3x4 operator*(const Matrix3x4& m, const Scalar k);
	friend Matrix3x4 operator+(const Matrix3x4& m1, const Matrix3x4& m2);
	friend Matrix3x4 operator-(const Matrix3x4& m1, const Matrix3x4& m2);
	friend Vector4 operator*(const Matrix3x4& m, const Vector4& v);
	friend Vector4 operator*(const Vector4& v, const Matrix3x4& m);
	friend Matrix3x4 operator*(const Matrix3x4& m1, const Matrix3x4& m2);
	friend bool operator==(const Matrix3x4& m1, const Matrix3x4& m2);

	/** @brief Assignment Operator */
	Matrix3x4& operator=(const Matrix3x4& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
		return *this;
	}

	/** @brief Get a column of the matrix as a vector
	*  @param i Column number 0 indexed */
	Vector4 getColumn(int i) const
	{
		return Vector4(m_el[0][i], m_el[1][i], m_el[2][i]);
	}

	/** @brief Get a row of the matrix as a vector
	*  @param i Row number 0 indexed */
	const Vector4& getRow(int i) const
	{
		return m_el[i];
	}

	/** @brief Get a mutable reference to a row of the matrix as a vector
	*  @param i Row number 0 indexed */
	Vector4& operator[](int i)
	{
		return m_el[i];
	}

	/** @brief Get a const reference to a row of the matrix as a vector
	*  @param i Row number 0 indexed */
	const Vector4& operator[](int i) const
	{
		return m_el[i];
	}

	/** @brief Multiply by the target matrix on the right
	*  @param m Rotation matrix to be applied
	* Equivilant to this = this * m */
	Matrix3x4& operator*=(const Matrix3x4& m)

	{
#ifdef USE_SSE
		Scalar4 rv00, rv01, rv02;
		Scalar4 rv10, rv11, rv12;
		Scalar4 rv20, rv21, rv22;
		Scalar4 mv0, mv1, mv2;

		rv02 = m_el[0].m_vec;
		rv12 = m_el[1].m_vec;
		rv22 = m_el[2].m_vec;

		mv0 = SIMD_AND(m[0].m_vec, SIMDFFF0MASK);
		mv1 = SIMD_AND(m[1].m_vec, SIMDFFF0MASK);
		mv2 = SIMD_AND(m[2].m_vec, SIMDFFF0MASK);

		// rv0
		rv00 = SIMD_SPLAT(rv02, 0);
		rv01 = SIMD_SPLAT(rv02, 1);
		rv02 = SIMD_SPLAT(rv02, 2);

		rv00 = SIMD_MUL(rv00, mv0);
		rv01 = SIMD_MUL(rv01, mv1);
		rv02 = SIMD_MUL(rv02, mv2);

		// rv1
		rv10 = SIMD_SPLAT(rv12, 0);
		rv11 = SIMD_SPLAT(rv12, 1);
		rv12 = SIMD_SPLAT(rv12, 2);

		rv10 = SIMD_MUL(rv10, mv0);
		rv11 = SIMD_MUL(rv11, mv1);
		rv12 = SIMD_MUL(rv12, mv2);

		// rv2
		rv20 = SIMD_SPLAT(rv22, 0);
		rv21 = SIMD_SPLAT(rv22, 1);
		rv22 = SIMD_SPLAT(rv22, 2);

		rv20 = SIMD_MUL(rv20, mv0);
		rv21 = SIMD_MUL(rv21, mv1);
		rv22 = SIMD_MUL(rv22, mv2);

		rv00 = SIMD_ADD(rv00, rv01);
		rv10 = SIMD_ADD(rv10, rv11);
		rv20 = SIMD_ADD(rv20, rv21);

		m_el[0].m_vec = SIMD_ADD(rv00, rv02);
		m_el[1].m_vec = SIMD_ADD(rv10, rv12);
		m_el[2].m_vec = SIMD_ADD(rv20, rv22);
#else    
		setValue(
			m.tdotx(m_el[0]), m.tdoty(m_el[0]), m.tdotz(m_el[0]),
			m.tdotx(m_el[1]), m.tdoty(m_el[1]), m.tdotz(m_el[1]),
			m.tdotx(m_el[2]), m.tdoty(m_el[2]), m.tdotz(m_el[2]));
#endif
		return *this;
	}

	/** @brief Adds by the target matrix on the right
	*  @param m matrix to be applied
	* Equivilant to this = this + m */
	Matrix3x4& operator+=(const Matrix3x4& m)
	{
#ifdef USE_SSE
		m_el[0].m_vec = SIMD_ADD(m_el[0].m_vec, m.m_el[0].m_vec);
		m_el[1].m_vec = SIMD_ADD(m_el[1].m_vec, m.m_el[1].m_vec);
		m_el[2].m_vec = SIMD_ADD(m_el[2].m_vec, m.m_el[2].m_vec);
#else
		setValue(
			m_el[0][0] + m.m_el[0][0],
			m_el[0][1] + m.m_el[0][1],
			m_el[0][2] + m.m_el[0][2],
			m_el[1][0] + m.m_el[1][0],
			m_el[1][1] + m.m_el[1][1],
			m_el[1][2] + m.m_el[1][2],
			m_el[2][0] + m.m_el[2][0],
			m_el[2][1] + m.m_el[2][1],
			m_el[2][2] + m.m_el[2][2]);
#endif
		return *this;
	}

	/** @brief Substractss by the target matrix on the right
	*  @param m matrix to be applied
	* Equivilant to this = this - m */
	Matrix3x4& operator-=(const Matrix3x4& m)
	{
#ifdef USE_SSE
		m_el[0].m_vec = SIMD_SUB(m_el[0].m_vec, m.m_el[0].m_vec);
		m_el[1].m_vec = SIMD_SUB(m_el[1].m_vec, m.m_el[1].m_vec);
		m_el[2].m_vec = SIMD_SUB(m_el[2].m_vec, m.m_el[2].m_vec);
#else
		setValue(
			m_el[0][0] - m.m_el[0][0],
			m_el[0][1] - m.m_el[0][1],
			m_el[0][2] - m.m_el[0][2],
			m_el[1][0] - m.m_el[1][0],
			m_el[1][1] - m.m_el[1][1],
			m_el[1][2] - m.m_el[1][2],
			m_el[2][0] - m.m_el[2][0],
			m_el[2][1] - m.m_el[2][1],
			m_el[2][2] - m.m_el[2][2]);
#endif
		return *this;
	}

	/** @brief Set from the rotational part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	/*void setFromOpenGLSubMatrix(const Scalar *m)
	{
		m_el[0].setValue(m[0], m[4], m[8]);
		m_el[1].setValue(m[1], m[5], m[9]);
		m_el[2].setValue(m[2], m[6], m[10]);

	}*/


	/**@brief Get the matrix represented as a quaternion
	* @param q The quaternion which will be set */
	void getRotation(Quaternion& q) const
	{
	//TODO SSE

		Scalar trace = m_el[0].x() + m_el[1].y() + m_el[2].z();

		Scalar temp[4];

		if (trace > Scalar(0.0))
		{
			Scalar s = thSqrt(trace + Scalar(1.0));
			temp[3] = (s * Scalar(0.5));
			s = Scalar(0.5) / s;

			temp[0] = ((m_el[2].y() - m_el[1].z()) * s);
			temp[1] = ((m_el[0].z() - m_el[2].x()) * s);
			temp[2] = ((m_el[1].x() - m_el[0].y()) * s);
		}
		else
		{
			int i = m_el[0].x() < m_el[1].y() ?
				(m_el[1].y() < m_el[2].z() ? 2 : 1) :
				(m_el[0].x() < m_el[2].z() ? 2 : 0);
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;

			Scalar s = thSqrt(m_el[i][i] - m_el[j][j] - m_el[k][k] + Scalar(1.0));
			temp[i] = s * Scalar(0.5);
			s = Scalar(0.5) / s;

			temp[3] = (m_el[k][j] - m_el[j][k]) * s;
			temp[j] = (m_el[j][i] + m_el[i][j]) * s;
			temp[k] = (m_el[k][i] + m_el[i][k]) * s;
		}
		q.setValue(temp[0], temp[1], temp[2], temp[3]);
	}

	/** @brief Set the matrix from a quaternion
	*  @param q The Quaternion to match */
	void setRotation(const Quaternion& q)
	{
		Scalar d = q.length2();
		//btFullAssert(d != Scalar(0.0));
		Scalar s = Scalar(2.0) / d;
		
		//TODO SSE

		Scalar xs = q.x() * s, ys = q.y() * s, zs = q.z() * s;
		Scalar wx = q.w() * xs, wy = q.w() * ys, wz = q.w() * zs;
		Scalar xx = q.x() * xs, xy = q.x() * ys, xz = q.x() * zs;
		Scalar yy = q.y() * ys, yz = q.y() * zs, zz = q.z() * zs;
		setValue(
			Scalar(1.0) - (yy + zz), xy - wz, xz + wy,
			xy + wz, Scalar(1.0) - (xx + zz), yz - wx,
			xz - wy, yz + wx, Scalar(1.0) - (xx + yy));
	}




	/** @brief Set the values of the matrix explicitly (row major)
	*  @param xx Top left
	*  @param xy Top Middle
	*  @param xz Top Right
	*  @param yx Middle Left
	*  @param yy Middle Middle
	*  @param yz Middle Right
	*  @param zx Bottom Left
	*  @param zy Bottom Middle
	*  @param zz Bottom Right*/
	void setValue(const Scalar& xx, const Scalar& xy, const Scalar& xz,
		const Scalar& yx, const Scalar& yy, const Scalar& yz,
		const Scalar& zx, const Scalar& zy, const Scalar& zz)
	{
		m_el[0].setValue(xx, xy, xz);
		m_el[1].setValue(yx, yy, yz);
		m_el[2].setValue(zx, zy, zz);
	}

	/** @brief Set the matrix from euler angles using YPR around YXZ respectively
	*  @param yaw Yaw about Y axis
	*  @param pitch Pitch about X axis
	*  @param roll Roll about Z axis
	*/
	void setEulerYPR(const Scalar& yaw, const Scalar& pitch, const Scalar& roll)
	{
		setEulerZYX(roll, pitch, yaw);
	}

	/** @brief Set the matrix from euler angles YPR around ZYX axes
	* @param eulerX Roll about X axis
	* @param eulerY Pitch around Y axis
	* @param eulerZ Yaw aboud Z axis
	*
	* These angles are used to produce a rotation matrix. The euler
	* angles are applied in ZYX order. I.e a vector is first rotated
	* about X then Y and then Z
	**/
	void setEulerZYX(Scalar eulerX, Scalar eulerY, Scalar eulerZ) {
		///@todo proposed to reverse this since it's labeled zyx but takes arguments xyz and it will match all other parts of the code
		Scalar ci(thCos(eulerX));
		Scalar cj(thCos(eulerY));
		Scalar ch(thCos(eulerZ));
		Scalar si(thSin(eulerX));
		Scalar sj(thSin(eulerY));
		Scalar sh(thSin(eulerZ));
		Scalar cc = ci * ch;
		Scalar cs = ci * sh;
		Scalar sc = si * ch;
		Scalar ss = si * sh;

		setValue(cj * ch, sj * sc - cs, sj * cc + ss,
			cj * sh, sj * ss + cc, sj * cs - sc,
			-sj, cj * si, cj * ci);
	}

	/** @brief Set from the rotational part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void setFromOpenGLMatrixRotationPart(Scalar *m)
	{
		m[0] = m_el[0][0];
		m[4] = m_el[0][1];
		m[8] = m_el[0][2];
		m[1] = m_el[1][0];
		m[5] = m_el[1][1];
		m[9] = m_el[1][2];
		m[2] = m_el[2][0];
		m[6] = m_el[2][1];
		m[10] = m_el[2][2];
	}

	/** @brief Fill the rotational part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void getOpenGLMatrixRotationPart(Scalar *m)
	{

		m_el[0].setValue(m[0], m[4], m[8]);
		m_el[1].setValue(m[1], m[5], m[9]);
		m_el[2].setValue(m[2], m[6], m[10]);
	}

	/**@brief Set the matrix to the identity */
	void setIdentity()
	{
#ifdef USE_SSE
		m_el[0] = SIMDV1000;
		m_el[1] = SIMDV0100;
		m_el[2] = SIMDV0010;
#else
		setValue(Scalar(1.0), Scalar(0.0), Scalar(0.0),
			Scalar(0.0), Scalar(1.0), Scalar(0.0),
			Scalar(0.0), Scalar(0.0), Scalar(1.0));
#endif
	}

	static const Matrix3x4&	getIdentity()
	{
#ifdef USE_SSE
		static const Matrix3x4
			identityMatrix(SIMDV1000, SIMDV0100, SIMDV0010);
#else
		static const Matrix3x4
			identityMatrix(
			Scalar(1.0), Scalar(0.0), Scalar(0.0),
			Scalar(0.0), Scalar(1.0), Scalar(0.0),
			Scalar(0.0), Scalar(0.0), Scalar(1.0));
#endif
		return identityMatrix;
	}

	/**@brief Get the matrix represented as euler angles around YXZ, roundtrip with setEulerYPR
	* @param yaw Yaw around Y axis
	* @param pitch Pitch around X axis
	* @param roll around Z axis */
	void getEulerYPR(Scalar& yaw, Scalar& pitch, Scalar& roll) const
	{

		// first use the normal calculus
		yaw = Scalar(thAtan2(m_el[1].x(), m_el[0].x()));
		pitch = Scalar(thAsin(-m_el[2].x()));
		roll = Scalar(thAtan2(m_el[2].y(), m_el[2].z()));

		// on pitch = +/-HalfPI
		if (thFabs(pitch) == TH_HALF_PI)
		{
			if (yaw>0)
				yaw -= TH_PI;
			else
				yaw += TH_PI;

			if (roll>0)
				roll -= TH_PI;
			else
				roll += TH_PI;
		}
	};


	/**@brief Get the matrix represented as euler angles around ZYX
	* @param yaw Yaw around X axis
	* @param pitch Pitch around Y axis
	* @param roll around X axis
	* @param solution_number Which solution of two possible solutions ( 1 or 2) are possible values*/
	void getEulerZYX(Scalar& yaw, Scalar& pitch, Scalar& roll, unsigned int solution_number = 1) const
	{
		struct Euler
		{
			Scalar yaw;
			Scalar pitch;
			Scalar roll;
		};

		Euler euler_out;
		Euler euler_out2; //second solution
		//get the pointer to the raw data

		// Check that pitch is not at a singularity
		if (thFabs(m_el[2].x()) >= 1)
		{
			euler_out.yaw = 0;
			euler_out2.yaw = 0;

			// From difference of angles formula
			Scalar delta = thAtan2(m_el[0].x(), m_el[0].z());
			if (m_el[2].x() > 0)  //gimbal locked up
			{
				euler_out.pitch = TH_PI / Scalar(2.0);
				euler_out2.pitch = TH_PI / Scalar(2.0);
				euler_out.roll = euler_out.pitch + delta;
				euler_out2.roll = euler_out.pitch + delta;
			}
			else // gimbal locked down
			{
				euler_out.pitch = -TH_PI / Scalar(2.0);
				euler_out2.pitch = -TH_PI / Scalar(2.0);
				euler_out.roll = -euler_out.pitch + delta;
				euler_out2.roll = -euler_out.pitch + delta;
			}
		}
		else
		{
			euler_out.pitch = -thAsin(m_el[2].x());
			euler_out2.pitch = TH_PI - euler_out.pitch;

			euler_out.roll = thAtan2(m_el[2].y() / thCos(euler_out.pitch),
				m_el[2].z() / thCos(euler_out.pitch));
			euler_out2.roll = thAtan2(m_el[2].y() / thCos(euler_out2.pitch),
				m_el[2].z() / thCos(euler_out2.pitch));

			euler_out.yaw = thAtan2(m_el[1].x() / thCos(euler_out.pitch),
				m_el[0].x() / thCos(euler_out.pitch));
			euler_out2.yaw = thAtan2(m_el[1].x() / thCos(euler_out2.pitch),
				m_el[0].x() / thCos(euler_out2.pitch));
		}

		if (solution_number == 1)
		{
			yaw = euler_out.yaw;
			pitch = euler_out.pitch;
			roll = euler_out.roll;
		}
		else
		{
			yaw = euler_out2.yaw;
			pitch = euler_out2.pitch;
			roll = euler_out2.roll;
		}
	}

	/**@brief Create a scaled copy of the matrix
	* @param s Scaling vector The elements of the vector will scale each column */

	Matrix3x4 scaled(const Vector4& s) const
	{
#ifdef USE_SSE
		return Matrix3x4(m_el[0] * s, m_el[1] * s, m_el[2] * s);
#else		
		return Matrix3x4(
			m_el[0].x() * s.x(), m_el[0].y() * s.y(), m_el[0].z() * s.z(),
			m_el[1].x() * s.x(), m_el[1].y() * s.y(), m_el[1].z() * s.z(),
			m_el[2].x() * s.x(), m_el[2].y() * s.y(), m_el[2].z() * s.z());
#endif
	}

	/**@brief Return the determinant of the matrix */
	Scalar determinant() const
	{
		return thTriple((*this)[0], (*this)[1], (*this)[2]);
	}
	
	/**@brief Return the adjoint of the matrix */
	Matrix3x4 adjoint() const
	{
		return Matrix3x4(cofac(1, 1, 2, 2), cofac(0, 2, 2, 1), cofac(0, 1, 1, 2),
			cofac(1, 2, 2, 0), cofac(0, 0, 2, 2), cofac(0, 2, 1, 0),
			cofac(1, 0, 2, 1), cofac(0, 1, 2, 0), cofac(0, 0, 1, 1));
	}
	
	/**@brief Return the matrix with all values non negative */
	Matrix3x4 absolute() const
	{
#ifdef USE_SSE
		return Matrix3x4(
			SIMD_AND(m_el[0].m_vec, SIMDABSMASK),
			SIMD_AND(m_el[1].m_vec, SIMDABSMASK),
			SIMD_AND(m_el[2].m_vec, SIMDABSMASK));
#else	
		return Matrix3x4(
			thFabs(m_el[0].x()), thFabs(m_el[0].y()), thFabs(m_el[0].z()),
			thFabs(m_el[1].x()), thFabs(m_el[1].y()), thFabs(m_el[1].z()),
			thFabs(m_el[2].x()), thFabs(m_el[2].y()), thFabs(m_el[2].z()));
#endif
	}

	//TODO
	/**@brief Return the transpose of the matrix */
	Matrix3x4 transpose() const
	{
		return Matrix3x4(m_el[0].x(), m_el[1].x(), m_el[2].x(),
			m_el[0].y(), m_el[1].y(), m_el[2].y(),
			m_el[0].z(), m_el[1].z(), m_el[2].z());
	}
	/**@brief Return the inverse of the matrix */
	Matrix3x4 inverse() const
	{
		Vector4 co(cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1));
		Scalar det = (*this)[0].dot(co);
		Scalar s = Scalar(1.0) / det;
		return Matrix3x4(co.x() * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s,
			co.y() * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s,
			co.z() * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
	}

	Matrix3x4 transposeTimes(const Matrix3x4& m) const
	{
#ifdef USE_SSE
		// zeros w
		//    static const __m128i xyzMask = (const __m128i){ -1ULL, 0xffffffffULL };
		Scalar4 row = m_el[0].m_vec;
		Scalar4 m0 = SIMD_AND(m.getRow(0).m_vec, SIMDFFF0MASK);
		Scalar4 m1 = SIMD_AND(m.getRow(1).m_vec, SIMDFFF0MASK);
		Scalar4 m2 = SIMD_AND(m.getRow(2).m_vec, SIMDFFF0MASK);
		Scalar4 r0 = SIMD_MUL(m0, SIMD_SHUFFLE(row, 0));
		Scalar4 r1 = SIMD_MUL(m0, SIMD_SHUFFLE(row, 0x55));
		Scalar4 r2 = SIMD_MUL(m0, SIMD_SHUFFLE(row, 0xaa));
		row = m_el[1].m_vec;
		r0 = SIMD_ADD(r0, SIMD_MUL(m1, SIMD_SHUFFLE(row, 0)));
		r1 = SIMD_ADD(r1, SIMD_MUL(m1, SIMD_SHUFFLE(row, 0x55)));
		r2 = SIMD_ADD(r2, SIMD_MUL(m1, SIMD_SHUFFLE(row, 0xaa)));
		row = m_el[2].m_vec;
		r0 = SIMD_ADD(r0, SIMD_MUL(m2, SIMD_SHUFFLE(row, 0)));
		r1 = SIMD_ADD(r1, SIMD_MUL(m2, SIMD_SHUFFLE(row, 0x55)));
		r2 = SIMD_ADD(r2, SIMD_MUL(m2, SIMD_SHUFFLE(row, 0xaa)));
		return Matrix3x4(r0, r1, r2);
#else
		return Matrix3x4(
			m_el[0].x() * m[0].x() + m_el[1].x() * m[1].x() + m_el[2].x() * m[2].x(),
			m_el[0].x() * m[0].y() + m_el[1].x() * m[1].y() + m_el[2].x() * m[2].y(),
			m_el[0].x() * m[0].z() + m_el[1].x() * m[1].z() + m_el[2].x() * m[2].z(),
			m_el[0].y() * m[0].x() + m_el[1].y() * m[1].x() + m_el[2].y() * m[2].x(),
			m_el[0].y() * m[0].y() + m_el[1].y() * m[1].y() + m_el[2].y() * m[2].y(),
			m_el[0].y() * m[0].z() + m_el[1].y() * m[1].z() + m_el[2].y() * m[2].z(),
			m_el[0].z() * m[0].x() + m_el[1].z() * m[1].x() + m_el[2].z() * m[2].x(),
			m_el[0].z() * m[0].y() + m_el[1].z() * m[1].y() + m_el[2].z() * m[2].y(),
			m_el[0].z() * m[0].z() + m_el[1].z() * m[1].z() + m_el[2].z() * m[2].z());
#endif
	}

	//TODO
	Matrix3x4 timesTranspose(const Matrix3x4& m) const
	{
		return Matrix3x4(
			m_el[0].dot(m[0]), m_el[0].dot(m[1]), m_el[0].dot(m[2]),
			m_el[1].dot(m[0]), m_el[1].dot(m[1]), m_el[1].dot(m[2]),
			m_el[2].dot(m[0]), m_el[2].dot(m[1]), m_el[2].dot(m[2]));
	}

	Scalar tdotx(const Vector4& v) const
	{
		return m_el[0].x() * v.x() + m_el[1].x() * v.y() + m_el[2].x() * v.z();
	}
	Scalar tdoty(const Vector4& v) const
	{
		return m_el[0].y() * v.x() + m_el[1].y() * v.y() + m_el[2].y() * v.z();
	}
	Scalar tdotz(const Vector4& v) const
	{
		return m_el[0].z() * v.x() + m_el[1].z() * v.y() + m_el[2].z() * v.z();
	}


	/**@brief diagonalizes this matrix by the Jacobi method.
	* @param rot stores the rotation from the coordinate system in which the matrix is diagonal to the original
	* coordinate system, i.e., old_this = rot * new_this * rot^T.
	* @param threshold See iteration
	* @param iteration The iteration stops when all off-diagonal elements are less than the threshold multiplied
	* by the sum of the absolute values of the diagonal, or when maxSteps have been executed.
	*
	* Note that this matrix is assumed to be symmetric.
	*/
	void diagonalize(Matrix3x4& rot, Scalar threshold, int maxSteps)
	{
		rot.setIdentity();
		for (int step = maxSteps; step > 0; step--)
		{
			// find off-diagonal element [p][q] with largest magnitude
			int p = 0;
			int q = 1;
			int r = 2;
			Scalar max = thFabs(m_el[0][1]);
			Scalar v = thFabs(m_el[0][2]);
			if (v > max)
			{
				q = 2;
				r = 1;
				max = v;
			}
			v = thFabs(m_el[1][2]);
			if (v > max)
			{
				p = 1;
				q = 2;
				r = 0;
				max = v;
			}

			Scalar t = threshold * (thFabs(m_el[0][0]) + thFabs(m_el[1][1]) + thFabs(m_el[2][2]));
			if (max <= t)
			{
				if (max <= TH_EPSILON * t)
				{
					return;
				}
				step = 1;
			}

			// compute Jacobi rotation J which leads to a zero for element [p][q] 
			Scalar mpq = m_el[p][q];
			Scalar theta = (m_el[q][q] - m_el[p][p]) / (2 * mpq);
			Scalar theta2 = theta * theta;
			Scalar cos;
			Scalar sin;
			if (theta2 * theta2 < Scalar(10 / TH_EPSILON))
			{
				t = (theta >= 0) ? 1 / (theta + thSqrt(1 + theta2))
					: 1 / (theta - thSqrt(1 + theta2));
				cos = 1 / thSqrt(1 + t * t);
				sin = cos * t;
			}
			else
			{
				// approximation for large theta-value, i.e., a nearly diagonal matrix
				t = 1 / (theta * (2 + Scalar(0.5) / theta2));
				cos = 1 - Scalar(0.5) * t * t;
				sin = cos * t;
			}

			// apply rotation to matrix (this = J^T * this * J)
			m_el[p][q] = m_el[q][p] = 0;
			m_el[p][p] -= t * mpq;
			m_el[q][q] += t * mpq;
			Scalar mrp = m_el[r][p];
			Scalar mrq = m_el[r][q];
			m_el[r][p] = m_el[p][r] = cos * mrp - sin * mrq;
			m_el[r][q] = m_el[q][r] = cos * mrq + sin * mrp;

			// apply rotation to rot (rot = rot * J)
			for (int i = 0; i < 3; i++)
			{
				Vector4& row = rot[i];
				mrp = row[p];
				mrq = row[q];
				row[p] = cos * mrp - sin * mrq;
				row[q] = cos * mrq + sin * mrp;
			}
		}
	}

	/**@brief Calculate the matrix cofactor
	* @param r1 The first row to use for calculating the cofactor
	* @param c1 The first column to use for calculating the cofactor
	* @param r1 The second row to use for calculating the cofactor
	* @param c1 The second column to use for calculating the cofactor
	* See http://en.wikipedia.org/wiki/Cofactor_(linear_algebra) for more details
	*/
	Scalar cofac(int r1, int c1, int r2, int c2) const
	{
		return m_el[r1][c1] * m_el[r2][c2] - m_el[r1][c2] * m_el[r2][c1];
	}

	void print()
	{
		std::cout << "[" << m_el[0].x() << ", " << m_el[0].y() << ", " << m_el[0].z() << "]" << std::endl;
		std::cout << "[" << m_el[1].x() << ", " << m_el[1].y() << ", " << m_el[1].z() << "]" << std::endl;
		std::cout << "[" << m_el[2].x() << ", " << m_el[2].y() << ", " << m_el[2].z() << "]" << std::endl;
	}

	std::string toString()
	{
		std::ostringstream os;
		os << "[" << m_el[0].x() << ", " << m_el[0].y() << ", " << m_el[0].z() << "]" << std::endl;
		os << "[" << m_el[1].x() << ", " << m_el[1].y() << ", " << m_el[1].z() << "]" << std::endl;
		os << "[" << m_el[2].x() << ", " << m_el[2].y() << ", " << m_el[2].z() << "]" << std::endl;
		return os.str();
	}
};

Matrix3x4 operator*(const Matrix3x4& m, const Scalar k)
{
#ifdef USE_SSE
	Scalar4 vk = SIMD_SET(k, k, k, 0);
	return Matrix3x4(
		SIMD_MUL(m[0].m_vec, vk),
		SIMD_MUL(m[1].m_vec, vk),
		SIMD_MUL(m[2].m_vec, vk));
#else
	return Matrix3x4(
		m[0].x()*k, m[0].y()*k, m[0].z()*k,
		m[1].x()*k, m[1].y()*k, m[1].z()*k,
		m[2].x()*k, m[2].y()*k, m[2].z()*k);
#endif
}

Matrix3x4 operator+(const Matrix3x4& m1, const Matrix3x4& m2)
{
#ifdef USE_SSE
	return Matrix3x4(
		SIMD_ADD(m1[0].m_vec, m2[0].m_vec),
		SIMD_ADD(m1[1].m_vec, m2[1].m_vec),
		SIMD_ADD(m1[2].m_vec, m2[2].m_vec));
#else
	return Matrix3x4(
		m1[0][0] + m2[0][0],
		m1[0][1] + m2[0][1],
		m1[0][2] + m2[0][2],

		m1[1][0] + m2[1][0],
		m1[1][1] + m2[1][1],
		m1[1][2] + m2[1][2],

		m1[2][0] + m2[2][0],
		m1[2][1] + m2[2][1],
		m1[2][2] + m2[2][2]);
#endif    
}

Matrix3x4 operator-(const Matrix3x4& m1, const Matrix3x4& m2)
{
#ifdef USE_SSE
	return Matrix3x4(
		SIMD_SUB(m1[0].m_vec, m2[0].m_vec),
		SIMD_SUB(m1[1].m_vec, m2[1].m_vec),
		SIMD_SUB(m1[2].m_vec, m2[2].m_vec));
#else
	return Matrix3x4(
		m1[0][0] - m2[0][0],
		m1[0][1] - m2[0][1],
		m1[0][2] - m2[0][2],

		m1[1][0] - m2[1][0],
		m1[1][1] - m2[1][1],
		m1[1][2] - m2[1][2],

		m1[2][0] - m2[2][0],
		m1[2][1] - m2[2][1],
		m1[2][2] - m2[2][2]);
#endif
}

//TODO
Vector4 operator*(const Matrix3x4& m, const Vector4& v)
{
	return Vector4(m[0].dot(v), m[1].dot(v), m[2].dot(v));
}


Vector4 operator*(const Vector4& v, const Matrix3x4& m)
{
#ifdef USE_SSE
	const Scalar4 vv = v.m_vec;

	Scalar4 c0 = SIMD_SPLAT(vv, 0);
	Scalar4 c1 = SIMD_SPLAT(vv, 1);
	Scalar4 c2 = SIMD_SPLAT(vv, 2);

	c0 = SIMD_MUL(c0, SIMD_AND(m[0].m_vec, SIMDFFF0MASK));
	c1 = SIMD_MUL(c1, SIMD_AND(m[1].m_vec, SIMDFFF0MASK));
	c0 = SIMD_ADD(c0, c1);
	c2 = SIMD_MUL(c2, SIMD_AND(m[2].m_vec, SIMDFFF0MASK));

	return Vector4(SIMD_ADD(c0, c2));
#else
	return Vector4(m.tdotx(v), m.tdoty(v), m.tdotz(v));
#endif
}

Matrix3x4 operator*(const Matrix3x4& m1, const Matrix3x4& m2)
{
#ifdef USE_SSE

	Scalar4 m10 = m1[0].m_vec;
	Scalar4 m11 = m1[1].m_vec;
	Scalar4 m12 = m1[2].m_vec;

	Scalar4 m2v = SIMD_AND(m2[0].m_vec, SIMDFFF0MASK);

	Scalar4 c0 = SIMD_SPLAT(m10, 0);
	Scalar4 c1 = SIMD_SPLAT(m11, 0);
	Scalar4 c2 = SIMD_SPLAT(m12, 0);

	c0 = SIMD_MUL(c0, m2v);
	c1 = SIMD_MUL(c1, m2v);
	c2 = SIMD_MUL(c2, m2v);

	m2v = SIMD_AND(m2[1].m_vec, SIMDFFF0MASK);

	Scalar4 c0_1 = SIMD_SPLAT(m10, 1);
	Scalar4 c1_1 = SIMD_SPLAT(m11, 1);
	Scalar4 c2_1 = SIMD_SPLAT(m12, 1);

	c0_1 = SIMD_MUL(c0_1, m2v);
	c1_1 = SIMD_MUL(c1_1, m2v);
	c2_1 = SIMD_MUL(c2_1, m2v);

	m2v = SIMD_AND(m2[2].m_vec, SIMDFFF0MASK);

	c0 = SIMD_ADD(c0, c0_1);
	c1 = SIMD_ADD(c1, c1_1);
	c2 = SIMD_ADD(c2, c2_1);

	m10 = SIMD_SPLAT(m10, 2);
	m11 = SIMD_SPLAT(m11, 2);
	m12 = SIMD_SPLAT(m12, 2);

	m10 = SIMD_MUL(m10, m2v);
	m11 = SIMD_MUL(m11, m2v);
	m12 = SIMD_MUL(m12, m2v);

	c0 = SIMD_ADD(c0, m10);
	c1 = SIMD_ADD(c1, m11);
	c2 = SIMD_ADD(c2, m12);

	return Matrix3x4(c0, c1, c2);
#else	
	return Matrix3x4(
		m2.tdotx(m1[0]), m2.tdoty(m1[0]), m2.tdotz(m1[0]),
		m2.tdotx(m1[1]), m2.tdoty(m1[1]), m2.tdotz(m1[1]),
		m2.tdotx(m1[2]), m2.tdoty(m1[2]), m2.tdotz(m1[2]));
#endif
}

/**@brief Equality operator between two matrices
* It will test all elements are equal.  */
bool operator==(const Matrix3x4& m1, const Matrix3x4& m2)
{
#ifdef USE_SSE

	Scalar4 c0, c1, c2;

	c0 = SIMD_EQUAL(m1[0].m_vec, m2[0].m_vec);
	c1 = SIMD_EQUAL(m1[1].m_vec, m2[1].m_vec);
	c2 = SIMD_EQUAL(m1[2].m_vec, m2[2].m_vec);

	c0 = SIMD_AND(c0, c1);
	c0 = SIMD_AND(c0, c2);

	return (0x7 == (0x7 & SIMD_MOVEMASK(c0)));
#else 
	return
		(m1[0][0] == m2[0][0] && m1[1][0] == m2[1][0] && m1[2][0] == m2[2][0] &&
		m1[0][1] == m2[0][1] && m1[1][1] == m2[1][1] && m1[2][1] == m2[2][1] &&
		m1[0][2] == m2[0][2] && m1[1][2] == m2[1][2] && m1[2][2] == m2[2][2]);
#endif
}

#endif //TH_MATRIX_H