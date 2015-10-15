/////////////////////////////////////////////////////////////////////////////////
///  Copyright (C), 2015-?, Xiaojian Huang
///  \file  Transform.h
///  \brief Transform module in SIMD library
///  \details Transform related functions and SSE and AVX instruction micro
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
#ifndef TH_TRANSFORM_H
#define TH_TRANSFORM_H

#include "Vector.h"
#include "Matrix.h"
#include "Quaternion.h"


/**@brief The Transform class supports rigid transforms with only translation and rotation and no scaling/shear.
*It can be used in combination with Vector4, Quaternion and Matrix3x4 linear algebra classes. */
ATTRIBUTE_ALIGNED(class) Transform{

	///Storage for the rotation
	Matrix3x4 m_basis;
	///Storage for the translation
	Vector4   m_origin;

public:

	/**@brief No initialization constructor */
	Transform() {}

	/**@brief Constructor from Quaternion (optional Vector4 )
	* @param q Rotation from quaternion
	* @param c Translation from Vector (default 0,0,0) */
	explicit inline Transform(const Quaternion& q,
		const Vector4& c = Vector4(Scalar(0), Scalar(0), Scalar(0)))
		: m_basis(q),
		m_origin(c)
	{}

	/**@brief Constructor from Matrix3x4 (optional Vector4)
	* @param b Rotation from Matrix
	* @param c Translation from Vector default (0,0,0)*/
	explicit inline Transform(const Matrix3x4& b,
		const Vector4& c = Vector4(Scalar(0), Scalar(0), Scalar(0)))
		: m_basis(b),
		m_origin(c)
	{}
	/**@brief Copy constructor */
	inline Transform(const Transform& other)
		: m_basis(other.m_basis),
		m_origin(other.m_origin)
	{
	}

	friend bool operator==(const Transform& t1, const Transform& t2);

	/**@brief Assignment Operator */
	inline Transform& operator=(const Transform& other)
	{
		m_basis = other.m_basis;
		m_origin = other.m_origin;
		return *this;
	}


	/**@brief Set the current transform as the value of the product of two transforms
	* @param t1 Transform 1
	* @param t2 Transform 2
	* This = Transform1 * Transform2 */
	inline void mult(const Transform& t1, const Transform& t2) {
		m_basis = t1.m_basis * t2.m_basis;
		m_origin = t1(t2.m_origin);
	}

	/*		void multInverseLeft(const Transform& t1, const Transform& t2) {
	Vector4 v = t2.m_origin - t1.m_origin;
	m_basis = btMultTransposeLeft(t1.m_basis, t2.m_basis);
	m_origin = v * t1.m_basis;
	}
	*/

	/**@brief Return the transform of the vector */
	inline Vector4 operator()(const Vector4& x) const
	{
		return x.dot3(m_basis[0], m_basis[1], m_basis[2]) + m_origin;
	}

	/**@brief Return the transform of the vector */
	inline Vector4 operator*(const Vector4& x) const
	{
		return (*this)(x);
	}

	/**@brief Return the transform of the Quaternion */
	inline Quaternion operator*(const Quaternion& q) const
	{
		return getRotation() * q;
	}

	/**@brief Return the basis matrix for the rotation */
	inline Matrix3x4&       getBasis()          { return m_basis; }
	/**@brief Return the basis matrix for the rotation */
	inline const Matrix3x4& getBasis()    const { return m_basis; }

	/**@brief Return the origin vector translation */
	inline Vector4&         getOrigin()         { return m_origin; }
	/**@brief Return the origin vector translation */
	inline const Vector4&   getOrigin()   const { return m_origin; }

	/**@brief Return a quaternion representing the rotation */
	Quaternion getRotation() const {
		Quaternion q;
		m_basis.getRotation(q);
		return q;
	}


	/**@brief Set from an array
	* @param m A pointer to a 15 element array (12 rotation(row major padded on the right by 1), and 3 translation */
	void setFromOpenGLMatrix(Scalar *m)
	{
		m_basis.setFromOpenGLMatrixRotationPart(m);
		m_origin.setFromOpenGLMatrixTranslationPart(m);
	}

	/**@brief Fill an array(OpenGL Matrix)
	* @param m A pointer to a 15 element array (12 rotation(row major padded on the right by 1), and 3 translation */
	void getOpenGLMatrix(Scalar *m)
	{
		m_basis.getOpenGLMatrixRotationPart(m);
		m_origin.getOpenGLMatrixTranslationPart(m);
	}

	/**@brief Set the translational element
	* @param origin The vector to set the translation to */
	inline void setOrigin(const Vector4& origin)
	{
		m_origin = origin;
	}

	inline Vector4 invXform(const Vector4& inVec) const
	{
		Vector4 v = inVec - m_origin;
		return (m_basis.transpose() * v);
	}


	/**@brief Set the rotational element by Matrix3x4 */
	inline void setBasis(const Matrix3x4& basis)
	{
		m_basis = basis;
	}

	/**@brief Set the rotational element by Quaternion */
	inline void setRotation(const Quaternion& q)
	{
		m_basis.setRotation(q);
	}


	/**@brief Set this transformation to the identity */
	void setIdentity()
	{
		m_basis.setIdentity();
		m_origin.setValue(Scalar(0.0), Scalar(0.0), Scalar(0.0));
	}

	/**@brief Multiply this Transform by another(this = this * another)
	* @param t The other transform */
	Transform& operator*=(const Transform& t)
	{
		m_origin += m_basis * t.m_origin;
		m_basis *= t.m_basis;
		return *this;
	}

	/**@brief Return the inverse of this transform */
	Transform inverse() const
	{
		Matrix3x4 inv = m_basis.transpose();
		return Transform(inv, inv * (-m_origin));
	}

	/**@brief Return the inverse of this transform times the other transform
	* @param t The other transform
	* return this.inverse() * the other */
	Transform inverseTimes(const Transform& t) const
	{
		Vector4 v = t.getOrigin() - m_origin;
		return Transform(m_basis.transposeTimes(t.m_basis),
			v * m_basis);
	}

	/**@brief Return the product of this transform and the other */
	Transform operator*(const Transform& t) const
	{
		return Transform(m_basis * t.m_basis,
			(*this)(t.m_origin));
	}

	/**@brief Return an identity transform */
	static const Transform&	getIdentity()
	{
		static const Transform identityTransform(Matrix3x4::getIdentity());
		return identityTransform;
	}

	void print()
	{
		m_basis.print();
		m_origin.print();
	}

	std::string toString()
	{
		return m_basis.toString() + m_origin.toString();
	}

};

/**@brief Test if two transforms have all elements equal */
inline bool operator==(const Transform& t1, const Transform& t2)
{
	return (t1.getBasis() == t2.getBasis() &&
		t1.getOrigin() == t2.getOrigin());
}


#endif //BT_TRANSFORM_H






