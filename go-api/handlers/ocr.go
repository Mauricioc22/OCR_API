package handlers

import (
    "bytes"
    "io"
    "mime/multipart"
    "net/http"

    "github.com/labstack/echo/v4"
)

func PredictPlate(c echo.Context) error {
    // Obtener el archivo del form-data
    file, err := c.FormFile("file")
    if err != nil {
        return c.String(http.StatusBadRequest, "No se recibió archivo")
    }
    src, err := file.Open()
    if err != nil {
        return err
    }
    defer src.Close()

    // Crear un buffer para el nuevo form-data
    var b bytes.Buffer
    writer := multipart.NewWriter(&b)
    fw, err := writer.CreateFormFile("file", file.Filename)
    if err != nil {
        return err
    }
    if _, err = io.Copy(fw, src); err != nil {
        return err
    }
    writer.Close()

    // Hacer la petición al microservicio Python
    resp, err := http.Post("http://localhost:8000/predict", writer.FormDataContentType(), &b)
    if err != nil {
        return c.String(http.StatusInternalServerError, "Error al conectar con microservicio")
    }
    defer resp.Body.Close()

    // Leer la respuesta y devolverla al cliente
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return err
    }
    return c.Blob(resp.StatusCode, "application/json", body)
}